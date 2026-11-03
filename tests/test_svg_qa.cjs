const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');

const script = path.resolve(__dirname, '../analysis/render_svg_qa.cjs');
const hash = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');

function fixture(t, svg) {
  const parent = path.resolve(os.tmpdir());
  const root = fs.mkdtempSync(path.join(parent, 'navsim-svg-qa-'));
  t.after(() => {
    const resolved = path.resolve(root);
    if (path.dirname(resolved) !== parent || !path.basename(resolved).startsWith('navsim-svg-qa-')) throw new Error('Unsafe test cleanup target');
    fs.rmSync(resolved, {recursive: true, force: true});
  });
  const source = path.join(root, 'fixture.svg'), receipt = path.join(root, 'receipt.json'), output = path.join(root, 'qa.png');
  fs.writeFileSync(source, svg);
  fs.writeFileSync(receipt, JSON.stringify({figures: {'fixture.svg': hash(source)}, scope: 'Test fixture only, not scientific evidence'}));
  return {source, receipt, output};
}

function run(row, modules = process.env.SVG_QA_NODE_MODULES || 'unused-for-early-rejections') {
  return spawnSync(process.execPath, [script, row.source, row.output, row.receipt, modules], {encoding: 'utf8', windowsHide: true});
}

test('refuses changed source binding without writing a raster', t => {
  const row = fixture(t, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>');
  fs.appendFileSync(row.source, '\n<!-- changed -->');
  const result = run(row);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /differs from its scientific render receipt/);
  assert.equal(fs.existsSync(row.output), false);
});

test('refuses external SVG references before loading rendering dependencies', t => {
  const row = fixture(t, '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.invalid/private.png"/></svg>');
  const result = run(row);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /External SVG reference refused/);
  assert.equal(fs.existsSync(row.output), false);
});

test('preserves an existing unrelated QA output', t => {
  const row = fixture(t, '<svg xmlns="http://www.w3.org/2000/svg"/>');
  fs.writeFileSync(row.output, 'unrelated fixture');
  const before = hash(row.output);
  const result = run(row);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /fresh PNG\/receipt paths/);
  assert.equal(hash(row.output), before);
});

test('actual independent render retains source hash and pending-review receipt', t => {
  if (!process.env.SVG_QA_NODE_MODULES) {t.skip('Explicit installed sharp node_modules path required for real render test'); return;}
  const row = fixture(t, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="#0072b2"/></svg>');
  const before = hash(row.source);
  const result = run(row);
  assert.equal(result.status, 0, result.stderr);
  const receipt = JSON.parse(fs.readFileSync(row.output + '.receipt.json', 'utf8'));
  assert.equal(receipt.source_sha256, before);
  assert.equal(hash(row.source), before);
  assert.equal(receipt.output_sha256, hash(row.output));
  assert.equal(receipt.scientific_receipt_sha256, hash(row.receipt));
  assert.equal(receipt.status, 'rasterization_completed_visual_review_pending');
  assert.equal(receipt.renderer, 'sharp/libvips/librsvg');
  assert.ok(receipt.width > 0 && receipt.height > 0);
});
