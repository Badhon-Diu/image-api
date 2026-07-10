'use strict';

// =============================================================================
//  STUDENT MARK IMAGE-EXTRACTION API  —  imageapi.js
//  Image-only version: audio pipeline, EJS mobile-upload session flow, and
//  Vercel-specific paths have all been removed. Built to run on Render.
//  Large images are resized/compressed with sharp before being sent to the
//  vision model, so big phone-camera photos don't slow down or break analysis.
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
  port: process.env.PORT || 3002, // Render injects PORT automatically
  modelscope: {
    baseURL: 'https://api-inference.modelscope.ai/v1',
    apiKey: process.env.MODELSCOPE_TOKEN,
    model: 'Qwen/Qwen3.5-35B-A3B',
  },
  imageBatchSize:5,      // how many images to send to the vision model concurrently
  maxImagesPerRequest: 20, // hard cap per API call
  visionTimeoutMs: 60_000,
  maxCacheEntries: 500,    // simple bound so the in-memory cache can't grow forever
  resize: {
    maxDimension: 1600,   // longest side, in px, after resize (upscaling never happens)
    jpegQuality: 85,      // output quality after sharp compresses to JPEG
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
  // Bound the cache so a long-running Render instance doesn't leak memory.
  if (imageCache.size >= CONFIG.maxCacheEntries) {
    const oldestKey = imageCache.keys().next().value;
    imageCache.delete(oldestKey);
  }
  imageCache.set(key, value);
}

// Resizes/compresses the image with sharp before it goes to the vision model.
// This keeps large phone-camera photos (often 4-12 MB) from slowing down or
// failing the vision API call. Falls back to the original buffer if sharp
// can't process the file for any reason (corrupt image, unsupported format).
async function preprocessImage(buffer, originalMimetype) {
  try {
    const resized = await sharp(buffer)
      .rotate() // apply EXIF orientation so rotated phone photos come out upright
      .resize({
        width: CONFIG.resize.maxDimension,
        height: CONFIG.resize.maxDimension,
        fit: 'inside',
        withoutEnlargement: true, // never upscale small images
      })
      .jpeg({ quality: CONFIG.resize.jpegQuality })
      .toBuffer();

    return { buffer: resized, mimetype: 'image/jpeg' };
  } catch (err) {
    console.warn(`[Image] sharp preprocessing failed (${err.message}), using original buffer`);
    return { buffer, mimetype: originalMimetype };
  }
}

async function analyzeImage(file, students = []) {
  // Cache key is based on the ORIGINAL bytes, so re-uploading the same photo
  // still hits the cache even though it gets resized fresh each time.
  const cacheKey = getCacheKey(file.buffer, students);
  if (imageCache.has(cacheKey)) {
    console.log(`[Image] Cache hit: ${file.originalname}`);
    return imageCache.get(cacheKey);
  }

  const originalSizeKb = (file.buffer.length / 1024).toFixed(0);
  const { buffer: processedBuffer, mimetype: processedMimetype } =
    await preprocessImage(file.buffer, file.mimetype);
  const processedSizeKb = (processedBuffer.length / 1024).toFixed(0);
  console.log(`[Image] ${file.originalname}: ${originalSizeKb}KB -> ${processedSizeKb}KB`);

  const base64  = processedBuffer.toString('base64');
  const dataUrl = `data:${processedMimetype};base64,${base64}`;
  const prompt  = buildImagePrompt(students);

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

    if (!resp.ok) {
      throw new Error(`ModelScope error ${resp.status}: ${responseText}`);
    }

    const h = resp.headers;
    console.log(`[ModelScope Quota] Daily: ${h.get('modelscope-ratelimit-requests-remaining')}/${h.get('modelscope-ratelimit-requests-limit')} | Model: ${h.get('modelscope-ratelimit-model-requests-remaining')}/${h.get('modelscope-ratelimit-model-requests-limit')}`);

    const data = JSON.parse(responseText);
    const rawOutput = data.choices?.[0]?.message?.content;
    if (!rawOutput || rawOutput.trim() === '') {
      throw new Error(`Vision model (${CONFIG.modelscope.model}) returned an empty response`);
    }

    cacheSet(cacheKey, rawOutput);
    return rawOutput;
  } finally {
    clear();
  }
}

function parseImageOutput(rawText) {
  if (!rawText || rawText.trim() === '') {
    throw new Error('Vision model output was empty');
  }
  let cleanJson = stripThinkingBlock(rawText);
  const fenceMatch = cleanJson.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) cleanJson = fenceMatch[1].trim();
  const objectMatch = cleanJson.match(/\{[\s\S]*\}/);
  if (objectMatch) cleanJson = objectMatch[0];
  const parsed = JSON.parse(cleanJson);
  const items = Array.isArray(parsed) ? parsed : [parsed];
  return items.map(item => ({
    'student id': item['student id'] || item.studentId || item.student_id || item.studentid || 'N/A',
    mark: normalizeMark(item.mark),
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
//  SECTION 6 — MULTER (in-memory, no disk writes needed for Render)
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
    batchSize: CONFIG.imageBatchSize,
    cacheSize: imageCache.size,
    resize: CONFIG.resize,
  });
});

// ── Image Analysis ──────────────────────────────────────────────────────────
// POST /api/analyze-images
//   multipart/form-data
//     images   -> one or more image files (field name "images", up to maxImagesPerRequest)
//     students -> optional JSON string: [{ "id": "232-15-241", "name": "Rahim" }, ...]
//   Response: [{ "student id": "232-15-241", "mark": 17 }, ...]  (one entry per image, in order)

app.post('/api/analyze-images', imageUpload.array('images', CONFIG.maxImagesPerRequest), async (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No images provided. Attach files under the "images" field.' });
  }

  let students = [];
  if (req.body.students) {
    try { students = JSON.parse(req.body.students); } catch { /* ignore bad JSON */ }
  }

  const batchSize = CONFIG.imageBatchSize;
  const allResults = [];

  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, i + batchSize);
    const batchResults = await Promise.all(
      batch.map(async (file) => {
        try {
          const rawOutput = await analyzeImage(file, students);
          return parseImageOutput(rawOutput);
        } catch (err) {
          console.error(`[Image] Failed for ${file.originalname}: ${err.message}`);
          return [{ 'student id': 'N/A', mark: 0 }];
        }
      })
    );
    allResults.push(...batchResults.flat());
  }

  return res.json(allResults);
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
  console.log(`✓ Image batch  : ${CONFIG.imageBatchSize} per batch`);
});

module.exports = app;
