#!/usr/bin/env node
/**
 * Web interface for the calibration frame measurement tool.
 * Accepts an uploaded photo, runs the Python engine, returns the
 * measurements and the annotated image.
 */

const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const REPO_ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(REPO_ROOT, 'output');
const UPLOAD_DIR = path.join(os.tmpdir(), 'hole-measurer-uploads');
const PORT = process.env.PORT || 3010;

// The engine runs through uv so OpenCV is available without a system install
const UV = process.env.UV_BIN || path.join(os.homedir(), '.local', 'bin', 'uv');

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 30 * 1024 * 1024 },
});

const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use('/output', express.static(OUTPUT_DIR));

app.post('/api/measure', upload.single('photo'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ ok: false, error: 'No photo received' });
  }

  // Keep the original extension so OpenCV recognises the format
  const ext = path.extname(req.file.originalname || '').toLowerCase() || '.jpg';
  const imagePath = req.file.path + ext;
  fs.renameSync(req.file.path, imagePath);

  const args = [
    'run', '--with', 'opencv-python-headless', 'python',
    path.join(REPO_ROOT, 'measure_object.py'),
    imagePath,
    '--json',
  ];
  const { width, height, border, convexHull, smooth } = req.body;
  if (width) args.push('--width', String(parseFloat(width)));
  if (height) args.push('--height', String(parseFloat(height)));
  if (convexHull === 'true') args.push('--convex-hull');
  if (smooth !== undefined && smooth !== '') {
    args.push('--smooth', String(parseFloat(smooth)));
  }

  execFile(UV, args, { timeout: 120000 }, (err, stdout, stderr) => {
    fs.rm(imagePath, { force: true }, () => {});

    // The engine prints its JSON result as the last line of stdout
    const lines = (stdout || '').trim().split('\n');
    let result = null;
    for (let i = lines.length - 1; i >= 0; i--) {
      try { result = JSON.parse(lines[i]); break; } catch (_) { /* keep looking */ }
    }

    if (!result) {
      console.error('Engine failed:', err, stderr);
      return res.status(500).json({
        ok: false,
        error: 'The measurement engine did not return a result - check the server log',
      });
    }
    if (!result.ok) {
      return res.json(result);
    }

    result.image_url = '/output/' + path.basename(result.output_image);
    delete result.output_image;
    if (result.dxf) {
      result.dxf_url = '/output/' + path.basename(result.dxf);
      delete result.dxf;
    }
    if (result.svg) {
      result.svg_url = '/output/' + path.basename(result.svg);
      delete result.svg;
    }
    res.json(result);
  });
});

app.listen(PORT, () => {
  console.log(`Hole Measurer running at http://localhost:${PORT}`);
});
