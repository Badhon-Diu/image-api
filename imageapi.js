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
//  SECTION 2 — CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const CONFIG = {
  port: process.env.PORT || 3002,
  modelscope: {
    baseURL: 'https://api-inference.modelscope.ai/v1',
    apiKey: process.env.MODELSCOPE_TOKEN,
    model: 'Qwen/Qwen3.5-35B-A3B',
  },
  concurrency: 8,               // Max images processed concurrently
  maxImagesPerRequest: 20,
  visionTimeoutMs: 45_000,      // Fail fast if API hangs (45s max)
  maxCacheEntries: 500,
  resize: {
    maxDimension: 2048,         
    fastQuality: 88,            // Fast encoding, still perfect for OCR
    fallbackQuality: 75,        // Only used if image is massive
    targetSizeBytes: 4 * 1024 * 1024, 
  },
};

if (!CONFIG.modelscope.apiKey) {
  console.warn('[WARN] MODELSCOPE_TOKEN is not set — /api/analyze-images will fail until it is configured.');
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
  }
  imageCache.set(key, value);
}

// ─────────────────────────────────────────────────────────────────────────────
//  BLAZING FAST IMAGE PREPROCESSING
// ─────────────────────────────────────────────────────────────────────────────

async function preprocessImage(buffer, originalMimetype) {
  try {
    const meta = await sharp(buffer).metadata();
    
    // 🚀 FAST PATH: If already a small, web-ready image, skip sharp entirely
    if (
      originalMimetype === 'image/jpeg' &&
      !meta.hasAlpha &&
      meta.width <= 2048 &&
      meta.height <= 2048 &&
      buffer.length <= CONFIG.resize.targetSizeBytes
    ) {
      return { buffer, mimetype: 'image/jpeg' };
    }

    let pipeline = sharp(buffer).rotate(); // Auto-orient based on EXIF
    
    // Flatten transparency (PNG -> White background)
    if (meta.hasAlpha) {
      pipeline = pipeline.flatten({ background: '#ffffff' });
    }
    
    // Only resize if larger than 2048px
    if (meta.width > 2048 || meta.height > 2048) {
      pipeline = pipeline.resize({
        width: 2048,
        height: 2048,
        fit: 'inside',
        withoutEnlargement: true,
      });
    }
    
    // Single fast encoding pass
    let processed = await pipeline.jpeg({ quality: CONFIG.resize.fastQuality }).toBuffer();
    
    // Fallback for insanely large images (rare)
    if (processed.length > CONFIG.resize.targetSizeBytes) {
      processed = await sharp(processed).jpeg({ quality: CONFIG.resize.fallbackQuality }).toBuffer();
    }
    
    return { buffer: processed, mimetype: 'image/jpeg' };
  } catch (err) {
    console.error(`[Image] sharp failed: ${err.message}. Falling back.`);
    return { buffer, mimetype: originalMimetype };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ANALYZE A SINGLE IMAGE
// ─────────────────────────────────────────────────────────────────────────────

async function analyzeImage(file, students = []) {
  const cacheKey = getCacheKey(file.buffer, students);
  if (imageCache.has(cacheKey)) {
    return imageCache.get(cacheKey);
  }

  const { buffer: processedBuffer, mimetype: processedMimetype } =
    await preprocessImage(file.buffer, file.mimetype);

  const base64  = processedBuffer.toString('base64');
  const dataUrl = `data:${processedMimetype};base64,${base64}`;
  const prompt  = buildImagePrompt(students);

  let attempt = 0;
  const maxRetries = 2;

  while (true) {
    const { signal, clear } = createTimeout(CONFIG.visionTimeoutMs);
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

      // Smart Retry on Rate Limit (429) or Server Error (5xx)
      if ((resp.status === 429 || resp.status >= 500) && attempt < maxRetries) {
        console.warn(`[ModelScope] Error ${resp.status}. Retrying in ${attempt + 1}s...`);
        await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
        attempt++;
        continue;
      }

      if (!resp.ok) {
        throw new Error(`ModelScope error ${resp.status}: ${responseText}`);
      }

      const data = JSON.parse(responseText);
      const rawOutput = data.choices?.[0]?.message?.content;
      
      if (!rawOutput || rawOutput.trim() === '') {
        throw new Error('Vision model returned an empty response');
      }

      cacheSet(cacheKey, rawOutput);
      return rawOutput;
    } catch (err) {
      clear();
      // Do not retry on timeouts or network errors, fail fast to save user time
      throw err;
    }
  }
}

function parseImageOutput(rawText) {
  if (!rawText || rawText.trim() === '') {
    throw new Error('Vision model output was empty');
  }
  let cleanJson = stripThinkingBlock(rawText);
  const objectMatch = cleanJson.match(/\{[\s\S]*\}/);
  if (objectMatch) cleanJson = objectMatch[0];
  
  try {
    const parsed = JSON.parse(cleanJson);
    const items = Array.isArray(parsed) ? parsed : [parsed];
    return items.map(item => ({
      'student id': item['student id'] || item.studentId || item.student_id || item.studentid || 'N/A',
      mark: normalizeMark(item.mark),
    }));
  } catch (e) {
    console.error("Failed to parse JSON:", cleanJson);
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
    cb(new Error('Unsupported image format: ' + file.mimetype));
  },
});

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 7 — EXPRESS APP
// ─────────────────────────────────────────────────────────────────────────────

const app = express();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

app.get('/', (_req, res) => {
  res.json({ status: 'ok', service: 'IntelliMarks Image API' });
});

app.get('/api/health', (_req, res) => {
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
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No images provided. Attach files under the "images" field.' });
  }

  let students = [];
  if (req.body.students) {
    try { students = JSON.parse(req.body.students); } catch { /* ignore */ }
  }

  const results = new Array(files.length);
  let currentIndex = 0;

  // 🚀 DYNAMIC WORKER POOL
  // Processes up to CONFIG.concurrency images at a time. As soon as one finishes,
  // the next one starts immediately. No waiting for slow batch members.
  async function worker() {
    while (currentIndex < files.length) {
      const myIndex = currentIndex++;
      const file = files[myIndex];
      try {
        const rawOutput = await analyzeImage(file, students);
        results[myIndex] = parseImageOutput(rawOutput);
      } catch (err) {
        console.error(`[Image] Failed for ${file.originalname}: ${err.message}`);
        results[myIndex] = [{ 'student id': 'N/A', mark: 0 }];
      }
    }
  }

  // Start workers
  const workers = [];
  const workerCount = Math.min(CONFIG.concurrency, files.length);
  for (let i = 0; i < workerCount; i++) {
    workers.push(worker());
  }

  await Promise.all(workers);

  return res.json(results.flat());
});

// ── Global Error Handler ────────────────────────────────────────────────────

app.use((err, _req, res, _next) => {
  const message = err.code === 'LIMIT_FILE_SIZE'
    ? 'File too large (max 10 MB per image)'
    : err.message || 'An unexpected error occurred';
  console.error('[Server] Unhandled error:', message);
  res.status(400).json({ error: message });
});

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 8 — START SERVER
// ─────────────────────────────────────────────────────────────────────────────

app.listen(CONFIG.port, '0.0.0.0', () => {
  console.log(`✓ Image API running on port ${CONFIG.port}`);
  console.log(`✓ Vision model : ${CONFIG.modelscope.model}`);
  console.log(`✓ Concurrency  : ${CONFIG.concurrency} parallel images`);
});

module.exports = app;
