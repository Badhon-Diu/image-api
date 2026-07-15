'use strict';

// =============================================================================
//  STUDENT MARK IMAGE-EXTRACTION API  —  imageapi.js
//  Optimized for maximum speed and concurrency. Built to run on Render.
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 1 — DEPENDENCIES
// ─────────────────────────────────────────────────────────────────────────────

const crypto  = require('crypto');
const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const sharp   = require('sharp');
require('dotenv').config();

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 1.5 — LOGGER
// ─────────────────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toISOString();
}

const log = {
  info:  (...args) => console.log(`[INFO  ${ts()}]`, ...args),
  warn:  (...args) => console.warn(`[WARN  ${ts()}]`, ...args),
  error: (...args) => console.error(`[ERROR ${ts()}]`, ...args),
  debug: (...args) => console.log(`[DEBUG ${ts()}]`, ...args),
};

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 2 — CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const CONFIG = {
  port: process.env.PORT || 3002,
  modelscope: {
    baseURL: 'https://api-inference.modelscope.ai/v1',
    apiKey: process.env.MODELSCOPE_TOKEN,
    model: 'Qwen/Qwen3.5-35B-A3B',
  },
  concurrency: 5,               // Max images processed concurrently
  maxImagesPerRequest: 20,
  visionTimeoutMs: 360_000,      // Fail fast if API hangs (45s max)
  maxCacheEntries: 500,
  resize: {
    maxDimension: 2048,         
    fastQuality: 88,            // Fast encoding, still perfect for OCR
    fallbackQuality: 75,        // Only used if image is massive
    targetSizeBytes: 4 * 1024 * 1024, 
  },
};

log.info('Booting IntelliMarks Image API...');
log.info('Config loaded:', JSON.stringify({
  port: CONFIG.port,
  model: CONFIG.modelscope.model,
  concurrency: CONFIG.concurrency,
  maxImagesPerRequest: CONFIG.maxImagesPerRequest,
  visionTimeoutMs: CONFIG.visionTimeoutMs,
  maxCacheEntries: CONFIG.maxCacheEntries,
  resize: CONFIG.resize,
}));

if (!CONFIG.modelscope.apiKey) {
  console.warn('[WARN] MODELSCOPE_TOKEN is not set — /api/analyze-images will fail until it is configured.');
  log.warn('MODELSCOPE_TOKEN is missing from environment variables. All vision requests will fail until this is set.');
} else {
  log.info('MODELSCOPE_TOKEN detected (length: ' + CONFIG.modelscope.apiKey.length + ')');
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 3 — UTILITY HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function createTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(timer) };
}

function normalizeMark(value) {
  if (typeof value === 'number' && !isNaN(value)) return value;
  if (value === null || value === undefined) return 0;
  const parsed = parseInt(String(value), 10);
  return isNaN(parsed) ? 0 : parsed;
}

function stripThinkingBlock(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 4 — IMAGE PROMPT
// ─────────────────────────────────────────────────────────────────────────────

const IMAGE_PROMPT_BASE = `
You are an OCR extraction tool. Look at this test paper image and extract exactly two values.

WHAT TO FIND:

1. Student ID
   Look for a field labeled any of: "Student ID", "ID Number", "ID No", "Roll No"
   Copy the value exactly as written, including hyphens (e.g. "232-15-241").

2. Obtained Mark / Score
   Look for the final awarded score. It may appear as:
   - A circled or boxed number at the top of the paper
   - The value in the "Total" row under the "Marks Obtained" column in a marks table
   - A number next to "Total Marks", "Score", or "Obtained"
   Extract it as a plain integer only (e.g. write 17, not "17/20").

STRICT OUTPUT RULES:
- After your thinking, output ONLY this exact JSON object. Nothing else. No explanation. No markdown. No backticks.
- Format: {"student id": "value here", "mark": number here}
- Example: {"student id": "232-15-290", "mark": 17}
- If a value cannot be found, use null for that field.
- The JSON must start with { and end with }
`.trim();

function buildImagePrompt(students = []) {
  if (!students || students.length === 0) return IMAGE_PROMPT_BASE;
  const list = students.map(s => `  ID: ${s.id}  Name: ${s.name || '(no name)'}`).join('\n');
  return IMAGE_PROMPT_BASE + `

KNOWN STUDENTS IN THIS CLASS:
 ${list}

If the ID in the image is partially visible or unclear, match it to the closest entry above.
If only a name is visible, look it up in the list and use that student's ID.`;
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 5 — IMAGE SERVICE (ModelScope vision model)
// ─────────────────────────────────────────────────────────────────────────────

const imageCache = new Map();

function getCacheKey(buffer, students = []) {
  const imgHash = crypto.createHash('md5').update(buffer).digest('hex');
  if (!students.length) return imgHash;
  const ctx = crypto.createHash('md5').update(JSON.stringify(students)).digest('hex').slice(0, 8);
  return `${imgHash}-${ctx}`;
}

function cacheSet(key, value) {
  if (imageCache.size >= CONFIG.maxCacheEntries) {
    const oldestKey = imageCache.keys().next().value;
    imageCache.delete(oldestKey);
    log.debug(`Cache full (${CONFIG.maxCacheEntries} entries). Evicted oldest key: ${oldestKey}`);
  }
  imageCache.set(key, value);
  log.debug(`Cache SET for key: ${key} (cache size now: ${imageCache.size})`);
}

// ─────────────────────────────────────────────────────────────────────────────
//  BLAZING FAST IMAGE PREPROCESSING
// ─────────────────────────────────────────────────────────────────────────────

async function preprocessImage(buffer, originalMimetype) {
  const startTime = Date.now();
  try {
    const meta = await sharp(buffer).metadata();
    log.debug(`Image metadata: mimetype=${originalMimetype}, width=${meta.width}, height=${meta.height}, hasAlpha=${meta.hasAlpha}, sizeBytes=${buffer.length}`);
    
    // 🚀 FAST PATH: If already a small, web-ready image, skip sharp entirely
    if (
      originalMimetype === 'image/jpeg' &&
      !meta.hasAlpha &&
      meta.width <= 2048 &&
      meta.height <= 2048 &&
      buffer.length <= CONFIG.resize.targetSizeBytes
    ) {
      log.debug(`Preprocess FAST PATH (no resize needed) — took ${Date.now() - startTime}ms`);
      return { buffer, mimetype: 'image/jpeg' };
    }

    let pipeline = sharp(buffer).rotate(); // Auto-orient based on EXIF
    
    // Flatten transparency (PNG -> White background)
    if (meta.hasAlpha) {
      pipeline = pipeline.flatten({ background: '#ffffff' });
      log.debug('Flattening transparency (alpha channel detected) to white background');
    }
    
    // Only resize if larger than 2048px
    if (meta.width > 2048 || meta.height > 2048) {
      pipeline = pipeline.resize({
        width: 2048,
        height: 2048,
        fit: 'inside',
        withoutEnlargement: true,
      });
      log.debug(`Resizing image from ${meta.width}x${meta.height} to fit within 2048x2048`);
    }
    
    // Single fast encoding pass
    let processed = await pipeline.jpeg({ quality: CONFIG.resize.fastQuality }).toBuffer();
    log.debug(`Encoded at quality=${CONFIG.resize.fastQuality}, resultSizeBytes=${processed.length}`);
    
    // Fallback for insanely large images (rare)
    if (processed.length > CONFIG.resize.targetSizeBytes) {
      log.warn(`Processed image still exceeds target size (${processed.length} bytes). Re-encoding at fallback quality=${CONFIG.resize.fallbackQuality}`);
      processed = await sharp(processed).jpeg({ quality: CONFIG.resize.fallbackQuality }).toBuffer();
      log.debug(`Fallback re-encode resultSizeBytes=${processed.length}`);
    }
    
    log.debug(`Preprocess complete — took ${Date.now() - startTime}ms, finalSizeBytes=${processed.length}`);
    return { buffer: processed, mimetype: 'image/jpeg' };
  } catch (err) {
    console.error(`[Image] sharp failed: ${err.message}. Falling back.`);
    log.error(`preprocessImage failed after ${Date.now() - startTime}ms: ${err.message}`, err.stack);
    return { buffer, mimetype: originalMimetype };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ANALYZE A SINGLE IMAGE
// ─────────────────────────────────────────────────────────────────────────────

async function analyzeImage(file, students = []) {
  const cacheKey = getCacheKey(file.buffer, students);
  if (imageCache.has(cacheKey)) {
    log.info(`Cache HIT for file "${file.originalname}" (key: ${cacheKey}) — skipping API call`);
    return imageCache.get(cacheKey);
  }
  log.info(`Cache MISS for file "${file.originalname}" (key: ${cacheKey}) — proceeding to preprocess + API call`);

  const { buffer: processedBuffer, mimetype: processedMimetype } =
    await preprocessImage(file.buffer, file.mimetype);

  const base64  = processedBuffer.toString('base64');
  const dataUrl = `data:${processedMimetype};base64,${base64}`;
  const prompt  = buildImagePrompt(students);
  log.debug(`Prompt built for "${file.originalname}" (students provided: ${students.length}), base64 length=${base64.length}`);

  let attempt = 0;
  const maxRetries = 2;

  while (true) {
    const { signal, clear } = createTimeout(CONFIG.visionTimeoutMs);
    const callStart = Date.now();
    log.info(`Calling ModelScope API for "${file.originalname}" (attempt ${attempt + 1}/${maxRetries + 1}) using model ${CONFIG.modelscope.model}`);
    try {
      const resp = await fetch(`${CONFIG.modelscope.baseURL}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${CONFIG.modelscope.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: CONFIG.modelscope.model,
          max_tokens: 1000,
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: dataUrl } },
            ],
          }],
          stream: false,
        }),
        signal,
      });

      const responseText = await resp.text();
      clear();
      log.info(`ModelScope responded for "${file.originalname}" with status ${resp.status} in ${Date.now() - callStart}ms`);

      // Smart Retry on Rate Limit (429) or Server Error (5xx)
      if ((resp.status === 429 || resp.status >= 500) && attempt < maxRetries) {
        console.warn(`[ModelScope] Error ${resp.status}. Retrying in ${attempt + 1}s...`);
        log.warn(`Retryable error ${resp.status} for "${file.originalname}". Waiting ${1000 * (attempt + 1)}ms before retry ${attempt + 2}/${maxRetries + 1}. Body: ${responseText.slice(0, 300)}`);
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        attempt++;
        continue;
      }

      if (!resp.ok) {
        log.error(`ModelScope returned non-OK status ${resp.status} for "${file.originalname}". Body: ${responseText.slice(0, 500)}`);
        throw new Error(`ModelScope error ${resp.status}: ${responseText}`);
      }

      const data = JSON.parse(responseText);
      const rawOutput = data.choices?.[0]?.message?.content;
      
      if (!rawOutput || rawOutput.trim() === '') {
        log.error(`Vision model returned empty content for "${file.originalname}"`);
        throw new Error('Vision model returned an empty response');
      }

      log.debug(`Raw model output for "${file.originalname}" (first 300 chars): ${rawOutput.slice(0, 300)}`);
      cacheSet(cacheKey, rawOutput);
      log.info(`Successfully analyzed "${file.originalname}" in ${Date.now() - callStart}ms (total attempts: ${attempt + 1})`);
      return rawOutput;
    } catch (err) {
      clear();
      // Do not retry on timeouts or network errors, fail fast to save user time
      if (err.name === 'AbortError') {
        log.error(`Request TIMEOUT for "${file.originalname}" after ${CONFIG.visionTimeoutMs}ms (attempt ${attempt + 1})`);
      } else {
        log.error(`Fatal error analyzing "${file.originalname}" (attempt ${attempt + 1}): ${err.message}`, err.stack);
      }
      throw err;
    }
  }
}

function parseImageOutput(rawText) {
  if (!rawText || rawText.trim() === '') {
    log.error('parseImageOutput received empty rawText');
    throw new Error('Vision model output was empty');
  }
  let cleanJson = stripThinkingBlock(rawText);
  const objectMatch = cleanJson.match(/\{[\s\S]*\}/);
  if (objectMatch) cleanJson = objectMatch[0];
  
  try {
    const parsed = JSON.parse(cleanJson);
    const items = Array.isArray(parsed) ? parsed : [parsed];
    const mapped = items.map(item => ({
      'student id': item['student id'] || item.studentId || item.student_id || item.studentid || 'N/A',
      mark: normalizeMark(item.mark),
    }));
    log.info(`Parsed output successfully: ${JSON.stringify(mapped)}`);
    return mapped;
  } catch (e) {
    console.error("Failed to parse JSON:", cleanJson);
    log.error(`JSON parse failure. Cleaned text was: ${cleanJson.slice(0, 500)} | Error: ${e.message}`);
    return [{ 'student id': 'N/A', mark: 0 }];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 6 — MULTER
// ─────────────────────────────────────────────────────────────────────────────

const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024, files: CONFIG.maxImagesPerRequest },
  fileFilter(_req, file, cb) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
    if (allowed.includes(file.mimetype)) return cb(null, true);
    log.warn(`Rejected file "${file.originalname}" — unsupported mimetype: ${file.mimetype}`);
    cb(new Error('Unsupported image format: ' + file.mimetype));
  },
});

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 7 — EXPRESS APP
// ─────────────────────────────────────────────────────────────────────────────

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Request logging middleware — logs every incoming request
app.use((req, _res, next) => {
  log.info(`Incoming ${req.method} ${req.originalUrl} from ${req.ip}`);
  next();
});

app.get('/', (_req, res) => {
  res.json({ status: 'ok', service: 'IntelliMarks Image API' });
});

app.get('/api/health', (_req, res) => {
  log.info(`Health check requested. Cache size: ${imageCache.size}`);
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    visionModel: CONFIG.modelscope.model,
    concurrency: CONFIG.concurrency,
    cacheSize: imageCache.size,
  });
});

// ── Image Analysis ──────────────────────────────────────────────────────────

app.post('/api/analyze-images', imageUpload.array('images', CONFIG.maxImagesPerRequest), async (req, res) => {
  const requestId = crypto.randomBytes(4).toString('hex');
  const requestStart = Date.now();
  log.info(`[req:${requestId}] /api/analyze-images called`);

  const files = req.files;
  if (!files || files.length === 0) {
    log.warn(`[req:${requestId}] No images provided in request`);
    return res.status(400).json({ error: 'No images provided. Attach files under the "images" field.' });
  }

  log.info(`[req:${requestId}] Received ${files.length} file(s): ${files.map(f => f.originalname).join(', ')}`);

  let students = [];
  if (req.body.students) {
    try {
      students = JSON.parse(req.body.students);
      log.info(`[req:${requestId}] Parsed ${students.length} known student(s) from request body`);
    } catch (e) {
      log.warn(`[req:${requestId}] Failed to parse "students" field from request body: ${e.message}`);
      /* ignore */
    }
  }

  const results = new Array(files.length);
  let currentIndex = 0;
  let completedCount = 0;

  // 🚀 DYNAMIC WORKER POOL
  // Processes up to CONFIG.concurrency images at a time. As soon as one finishes,
  // the next one starts immediately. No waiting for slow batch members.
  async function worker(workerId) {
    while (currentIndex < files.length) {
      const myIndex = currentIndex++;
      const file = files[myIndex];
      log.debug(`[req:${requestId}] Worker ${workerId} picked up file[${myIndex}]: "${file.originalname}"`);
      try {
        const rawOutput = await analyzeImage(file, students);
        results[myIndex] = parseImageOutput(rawOutput);
        completedCount++;
        log.info(`[req:${requestId}] Worker ${workerId} completed file[${myIndex}] "${file.originalname}" (${completedCount}/${files.length} done)`);
      } catch (err) {
        console.error(`[Image] Failed for ${file.originalname}: ${err.message}`);
        log.error(`[req:${requestId}] Worker ${workerId} FAILED on file[${myIndex}] "${file.originalname}": ${err.message}`);
        results[myIndex] = [{ 'student id': 'N/A', mark: 0 }];
        completedCount++;
      }
    }
    log.debug(`[req:${requestId}] Worker ${workerId} finished — no more files left`);
  }

  // Start workers
  const workers = [];
  const workerCount = Math.min(CONFIG.concurrency, files.length);
  log.info(`[req:${requestId}] Spinning up ${workerCount} worker(s) for ${files.length} file(s)`);
  for (let i = 0; i < workerCount; i++) {
    workers.push(worker(i + 1));
  }

  await Promise.all(workers);

  const totalTime = Date.now() - requestStart;
  log.info(`[req:${requestId}] All files processed in ${totalTime}ms. Sending response.`);

  return res.json(results.flat());
});

// ── Global Error Handler ────────────────────────────────────────────────────

app.use((err, req, res, _next) => {
  const message = err.code === 'LIMIT_FILE_SIZE'
    ? 'File too large (max 10 MB per image)'
    : err.message || 'An unexpected error occurred';
  console.error('[Server] Unhandled error:', message);
  log.error(`Unhandled error on ${req.method} ${req.originalUrl}: ${message}`, err.stack);
  res.status(400).json({ error: message });
});

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 8 — START SERVER
// ─────────────────────────────────────────────────────────────────────────────

app.listen(CONFIG.port, '0.0.0.0', () => {
  console.log(`✓ Image API running on port ${CONFIG.port}`);
  console.log(`✓ Vision model : ${CONFIG.modelscope.model}`);
  console.log(`✓ Concurrency  : ${CONFIG.concurrency} parallel images`);
  log.info(`Server successfully started and listening on 0.0.0.0:${CONFIG.port}`);
});

// Catch-all process level logging so nothing fails silently
process.on('unhandledRejection', (reason) => {
  log.error('UNHANDLED PROMISE REJECTION:', reason);
});
process.on('uncaughtException', (err) => {
  log.error('UNCAUGHT EXCEPTION:', err.message, err.stack);
});

module.exports = app;
