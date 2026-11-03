// Independently rasterize an owned, receipt-bound SVG without changing its source.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const hash = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');

async function main() {
  const [sourceArg, outputArg, receiptArg, modulesArg] = process.argv.slice(2);
  if (!sourceArg || !outputArg || !receiptArg || !modulesArg || process.argv.length !== 6) {
    throw new Error('Usage: node render_svg_qa.cjs source.svg output.png scientific_receipt.json node_modules');
  }
  const source = path.resolve(sourceArg), output = path.resolve(outputArg), sourceReceipt = path.resolve(receiptArg);
  const receiptPath = output + '.receipt.json';
  if (path.extname(source).toLowerCase() !== '.svg' || path.extname(output).toLowerCase() !== '.png'
      || source === output || fs.existsSync(output) || fs.existsSync(receiptPath)) {
    throw new Error('Expected a source SVG and fresh PNG/receipt paths; no source/QA overwrite');
  }
  const originalHash = hash(source), receiptHash = hash(sourceReceipt);
  const scientific = JSON.parse(fs.readFileSync(sourceReceipt, 'utf8'));
  if (scientific.figures[path.basename(source)] !== originalHash) throw new Error('SVG differs from its scientific render receipt');
  const svg = fs.readFileSync(source, 'utf8');
  if (/(?:href|xlink:href)\s*=\s*["'](?!#|data:)[^"']+/i.test(svg)) throw new Error('External SVG reference refused');
  const sharp = require(path.join(path.resolve(modulesArg), 'sharp'));
  sharp.concurrency(1);
  fs.mkdirSync(path.dirname(output), {recursive: true});
  const temporary = output + '.tmp.png';
  if (fs.existsSync(temporary)) throw new Error('Preserve existing unfinished rasterization; use another QA output');
  const info = await sharp(source, {density: 144, limitInputPixels: 20000000}).png().toFile(temporary);
  if (hash(source) !== originalHash || hash(sourceReceipt) !== receiptHash) throw new Error('Source/receipt changed during independent render');
  fs.renameSync(temporary, output);
  const receipt = {status: 'rasterization_completed_visual_review_pending', source_path: source, source_sha256: originalHash,
                   scientific_receipt_path: sourceReceipt, scientific_receipt_sha256: receiptHash,
                   output_path: output, output_sha256: hash(output), density_dpi: 144, width: info.width, height: info.height,
                   renderer_source_sha256: hash(__filename), renderer: 'sharp/libvips/librsvg', versions: sharp.versions,
                   source_scope: scientific.scope, scope: 'Independent SVG layout QA only, not additional experimental evidence',
                   captured_utc: new Date().toISOString()};
  fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n', {flag: 'wx'});
  process.stdout.write(JSON.stringify({output, width: info.width, height: info.height, receipt: receiptPath}) + '\n');
}

main().catch(error => {process.stderr.write(error.stack + '\n'); process.exitCode = 1;});
