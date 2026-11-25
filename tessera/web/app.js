// Tessera v2 interface: four questions, one stage, one button.
import { Stage } from './stage.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const nf = (d) => new Intl.NumberFormat('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d });
const fmt0 = nf(0), fmt1 = nf(1);
const body = document.body;

let py = null;
let S = null;                 // state from Python
let built = null;             // last numbers
let view = 'art';              // 'art': the finished piece, 'plates': the build plates
let plateIdx = null;
let toastTimer = null;

const call = (name, ...args) => new Promise((res) => py[name](...args, (r) => res(r)));

// ------------------------------------------------------------------ stage
function makeStage() {
  try {
    return new Stage($('#canvas-host'), { onPick: (m) => showPart(m), onReady: () => {} });
  } catch (e) {
    // no WebGL on this machine/session: the app still works with the flat relit preview
    body.classList.add('no3d');
    const noop = () => {};
    return new Proxy({ az: -130, el: 40, renderer: { domElement: document.createElement('canvas') } }, {
      get: (t, k) => (k in t ? t[k] : noop),
      set: (t, k, v) => { t[k] = v; return true; },
    });
  }
}
const stage = makeStage();
window.stage = stage;

// ------------------------------------------------------------------ render state
function render() {
  if (!S) return;
  body.classList.toggle('empty', !S.image.path);
  // 1
  $('#src-thumb').src = S.image.url || '';
  $('#src-name').textContent = S.image.name;
  const win = $('#in-width');
  if (document.activeElement !== win) win.value = fmt0.format(S.width_cm).replace(/\./g, '');
  $('#btn-example').style.display = S.example ? '' : 'none';
  $('#btn-save').disabled = !S.image.path;
  renderFit();
  // 2
  renderColours();
  // 3
  const st = $('#styles');
  if (st.children.length !== S.styles.length) {
    st.innerHTML = '';
    for (const c of S.styles) {
      const b = document.createElement('button');
      b.className = 'style';
      b.dataset.style = c.name;
      b.title = c.aciklama || '';
      b.innerHTML = `<span class="th"><img alt=""><em></em></span><span class="nm">${esc(c.name)}</span>`;
      b.onclick = () => { if (S.image.path) py.setStyle(c.name); };
      st.appendChild(b);
    }
  }
  for (const b of st.children) {
    const name = b.dataset.style;
    b.classList.toggle('on', name === S.style);
    const img = b.querySelector('img');
    const url = S.thumbs[name];
    if (url && img.getAttribute('src') !== url) { img.onload = () => img.classList.add('ok'); img.src = url; }
    if (!url) { img.classList.remove('ok'); img.removeAttribute('src'); }
    b.querySelector('em').textContent = url || !S.image.path ? '' : '···';
  }
  // 4
  for (const m of $$('.mode')) m.classList.toggle('on', m.dataset.mode === S.mode);
  // fine
  renderFine();
  $('#btn-export').disabled = !S.image.path || !S.palette.length;
  renderLabel();
}

function renderHeight() {
  const o = $('#out-height');
  if (built && S && S.image.path) o.textContent = `× ${fmt0.format(built.size_mm[1] / 10)} cm yükseklik`;
  else o.textContent = '';
}

function renderNumbers() {
  renderHeight();
  if (!built) { $('#n-parca').innerHTML = '&nbsp;'; $('#n-ams').innerHTML = '&nbsp;'; return; }
  const p = built.estimates.parca || {}, a = built.estimates.ams || {};
  $('#n-parca').textContent = p.parca != null
    ? `${fmt0.format(p.parca)} taş · ${p.tabla} tabla · ~${fmt0.format(p.saat)} saat` : '';
  $('#n-ams').textContent = a.panel != null
    ? `${a.panel} panel · ~${fmt0.format(a.purge_gram)} g atık · ~${fmt0.format(a.saat)} saat` + ((S.palette || []).length > 4 ? ` · ${S.palette.length} renk: ${Math.ceil(S.palette.length / 4)} AMS gerekir` : '') : '';
}

function summaryLine() {
  if (!built || !S) return '';
  const e = built.estimates[S.mode] || {};
  const [w, h] = built.size_mm;
  const stones = (built.estimates.parca || {}).parca || built.tiles || 0;
  const parts = S.mode === 'ams' ? `${fmt0.format(stones)} taş, ${e.panel} panel` : `${fmt0.format(stones)} taş`;
  return `${fmt0.format(w / 10)} × ${fmt0.format(h / 10)} cm · ${parts} · ${e.tabla} tabla · ~${fmt0.format(e.saat || 0)} saat · ${fmt1.format((e.gram || 0) / 1000)} kg`;
}

function renderLabel() {
  if (!S) return;
  $('#label-t').textContent = S.name || S.image.name || '';
  $('#summary').textContent = body.classList.contains('busy') && !built ? 'Hesaplanıyor…' : summaryLine();
}

// ------------------------------------------------------------------ fine settings sheet
function renderFine() {
  const f = S.fine;
  const set = (id, v) => { const el = $(id); if (document.activeElement !== el) el.value = v; };
  for (const k of ['tas', 'derz', 'kalinlik', 'kabartma', 'oynama', 'panel']) { set('#f-' + k, f[k]); fineLabel(k, f[k]); }
  body.classList.toggle('mode-ams', S.mode === 'ams');
  $('#f-bolge').checked = !!f.otomatik_bolge;
  $('#f-acilis').checked = S.acilis !== false;
  segOn('#f-tabla', f.tabla_sirasi);
  segOn('#f-tram', String(f.tram));
  colourSeg('#f-taban', 'taban', 'Kendisi seçsin');
}
function fineLabel(k, v) {
  v = parseFloat(v);
  const t = { tas: `%${fmt0.format(v * 100)}`, panel: `${fmt0.format(v)} mm` }[k]
    || (v === 0 ? 'kapalı' : `${fmt1.format(v)} mm`);
  $(`#f-${k}-v`).textContent = t;
}
function colourSeg(sel, key, first) {
  const k = $(sel);
  k.innerHTML = '';
  const cur = S.fine[key] == null ? null : S.fine[key];
  const opts = [{ v: null, t: first }, ...S.palette.map((n) => ({ v: S.hexes[n], t: n, hex: S.hexes[n] }))];
  for (const o of opts) {
    const b = document.createElement('button');
    b.innerHTML = (o.hex ? `<span class="sw" style="background:${o.hex}"></span>` : '') + esc(o.t);
    b.classList.toggle('on', cur === o.v);
    b.onclick = () => fine({ [key]: o.v });
    k.appendChild(b);
  }
}
function segOn(sel, v) { for (const b of $(sel).children) b.classList.toggle('on', b.dataset.v === v); }
function fine(patch) { py.setFine(JSON.stringify(patch)); }
function bindFine() {
  for (const k of ['tas', 'derz', 'kalinlik', 'kabartma', 'oynama', 'panel']) {
    const el = $('#f-' + k);
    el.addEventListener('input', () => fineLabel(k, el.value));
    el.addEventListener('change', () => fine({ [k]: parseFloat(el.value) }));
  }
  $('#f-bolge').onchange = (e) => fine({ otomatik_bolge: e.target.checked });
  const seg = (sel, key, parse) => $(sel).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) fine({ [key]: parse(b.dataset.v) });
  });
  seg('#f-tabla', 'tabla_sirasi', (v) => v);
  seg('#f-tram', 'tram', (v) => (v === 'auto' ? 'auto' : v === 'true'));
  $('#f-acilis').onchange = (e) => py.setSplash(e.target.checked);
  $('#f-tohum').onclick = () => fine({ tohum: Math.floor(Math.random() * 9000) + 10 });
  $('#fine-reset').onclick = () => py.resetFine();
  $('#btn-fine').onclick = () => sheet('#sheet-fine');
  $('#fine-close').onclick = () => sheet(null);
}

function sheet(sel) {
  for (const s of $$('.sheet')) s.classList.toggle('on', sel && s.matches(sel) && !s.classList.contains('on'));
  body.classList.toggle('fine-open', $('#sheet-fine').classList.contains('on'));
}

// ------------------------------------------------------------------ colours of the picture
// what the program thinks of this picture, and the way out when it is wrong
function renderFit() {
  const u = S.uygun || {}, f = $('#fit');
  const on = !!(S.image.path && u.soz && u.soz.length && !S.finding);
  f.className = 'fit' + (on ? ' on ' + u.puan : '');
  if (!on) return;
  $('#fit-t').textContent = u.soz.join(' ');
  const simple = u.tur === 'ton';
  $('#fit-b').textContent = simple ? 'Sadeleştirmeden dene' : 'Sadeleştirerek dene';
  $('#fit-b').onclick = () => { openCol = null; py.setKind(simple ? 'net' : 'ton'); };
}
let openCol = null;
function renderColours() {
  const cs = S.colours || [];
  const host = $('#cols');
  host.innerHTML = '';
  if (openCol != null && openCol >= cs.length) openCol = null;
  $('#found').textContent = !S.image.path ? 'Bir görsel seçince renkleri burada görünür.'
    : S.finding ? 'Görselin renkleri bulunuyor…'
    : !cs.length ? 'Renkler bulunamadı. Baştan bulmayı dene.'
      : (S.uygun && S.uygun.tur === 'ton' ? `Görsel ${cs.length} renge sadeleştirildi.` : S.fine.renk_sayisi ? `Görsel ${cs.length} renge ayrıldı.` : `Görselde ${cs.length} renk var.`) + " Değiştirmek istediğine dokun.";
  cs.forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'col' + (openCol === i ? ' on' : '');
    b.style.setProperty('--c', c.hex);
    b.style.setProperty('--k', c.kaynak);
    b.title = 'Köşedeki ton, bu rengin görseldeki hâli';
    const pc = c.pay * 100;
    b.innerHTML = `<span class="sw"></span><span class="nm">${esc(c.ad)}</span><span class="pc">%${pc < 1 ? fmt1.format(Math.max(pc, 0.1)) : fmt0.format(pc)}</span>`;
    b.onclick = () => { openCol = openCol === i ? null : i; renderColours(); };
    host.appendChild(b);
  });
  $('#n-cols').textContent = cs.length || '';
  $('#n-less').disabled = S.finding || cs.length <= 2;
  $('#n-more').disabled = S.finding || cs.length >= 12 || !cs.length;
  $('#btn-recolour').style.visibility = !S.finding && (!cs.length || cs.some((c) => c.elle) || S.fine.renk_sayisi) ? 'visible' : 'hidden';
  renderPop();
}
function renderPop() {
  const p = $('#pop'), c = openCol == null ? null : S.colours[openCol];
  p.classList.toggle('on', !!c);
  if (!c) return;
  const i = openCol;
  $('#pop-t').textContent = `${c.ad} yerine`;
  const g = $('#pop-g');
  g.innerHTML = '';
  for (const m of S.main) {
    const b = document.createElement('button');
    b.style.setProperty('--c', m.hex);
    b.title = m.name;
    b.className = m.hex.toUpperCase() === c.hex.toUpperCase() ? 'on' : '';
    b.onclick = () => py.setColour(i, m.name);
    b.onmouseenter = () => { $('#pop-t').textContent = `${c.ad} yerine ${m.name.toLocaleLowerCase('tr')}`; };
    b.onmouseleave = () => { $('#pop-t').textContent = `${c.ad} yerine`; };
    g.appendChild(b);
  }
  $('#pop-same').style.setProperty('--c', c.kaynak);
  $('#pop-same').onclick = () => py.setColour(i, 'gorsel');
  $('#pop-c').value = c.hex.toLowerCase();
  $('#pop-c').onchange = (e) => py.setColour(i, e.target.value);
}

// ------------------------------------------------------------------ views
function setView(v) {
  view = v;
  for (const b of $$('#views button')) b.classList.toggle('on', b.dataset.view === v);
  body.classList.toggle('view-plates', v === 'plates');
  if (v === 'plates') {
    showPart(null);
    stage.select(null, true);
    call('getPlates').then((r) => { if (r) plates(JSON.parse(r)); });
  }
  stage.setView(v);
}

let flatTimer = null;             // the flat picture stands in where 3D cannot run
function loadFlat() {
  clearTimeout(flatTimer);
  flatTimer = setTimeout(async () => {
    const url = await call('getPreview');
    if (url) $('#flat-img').src = url;
  }, 60);
}

let platesState = null;
let plateOf = {};                 // stone code -> plate title (from plates_glb)
async function plates(info) {
  platesState = info;
  await stage.loadPlates(info.url, info.meta);
  body.classList.add('plates-ready');
  plateIdx = null;
  renderPlateNav();
}
function renderPlateNav() {
  if (!platesState) return;
  const ps = platesState.meta.plates || [];
  const t = $('#pl-t');
  if (plateIdx == null) {
    const g = ps.reduce((a, p) => a + (p.grams || 0), 0), h = ps.reduce((a, p) => a + (p.hours || 0), 0);
    t.innerHTML = `<b>${ps.length} tabla</b>~${fmt0.format(h)} saat · ${fmt1.format(g / 1000)} kg`;
  } else {
    const p = ps[plateIdx];
    const dot = p.hex ? `<i style="background:${p.hex}"></i>` : '';
    t.innerHTML = `<b>${plateIdx + 1} / ${ps.length}</b>${dot}${esc(p.title)} · ${p.codes.length} taş · ~${fmt1.format(p.hours)} saat`;
  }
}
function stepPlate(d) {
  const n = (platesState && platesState.meta.plates || []).length;
  if (!n) return;
  plateIdx = plateIdx == null ? (d > 0 ? 0 : n - 1) : (plateIdx + d + n) % n;
  stage.focusPlate(plateIdx);
  renderPlateNav();
}

// ------------------------------------------------------------------ part card
function showPart(m) {
  body.classList.toggle('picked', !!m);
  if (!m) return;
  $('#part-code').textContent = m.code;
  $('#part-sw').style.background = m.hex;
  $('#part-colour').textContent = m.colour_name;
  const [w, h, d] = m.size_mm;
  $('#part-size').textContent = `${fmt0.format(w)} × ${fmt0.format(h)} mm, ${fmt1.format(d)} mm yükseklik`;
  $('#part-panel').textContent = m.template_page ? `sayfa ${m.template_page} · panel ${m.panel}` : m.panel;
  $('#part-plate').textContent = m.plate || plateOf[m.code] || 'tablalar diziliyor…';
}

// ------------------------------------------------------------------ converting overlay
let convShown = 0, convTarget = 0, convTimer = null, convStart = 0;
function converting(p) {
  // p.f: 0..1 from real stages. Inside the long "stones" stage the number creeps slowly towards
  // the next milestone so it never looks frozen, but it never passes what has really been done.
  const on = p.f < 1;
  if (on && !body.classList.contains('is-converting')) { convShown = 0; convStart = performance.now(); }
  body.classList.toggle('is-converting', on);
  if (p.msg) $('#conv-msg').textContent = p.msg;
  convTarget = Math.max(p.f, on ? convTarget : 1);
  if (p.f === 0) convTarget = 0;
  if (!on) { convTarget = 1; convShown = 1; paintConv(); clearInterval(convTimer); convTimer = null; return; }
  convShown = Math.max(convShown, convTarget);
  paintConv();
  if (!convTimer) convTimer = setInterval(() => {
    const ceil = Math.min(0.99, convTarget + 0.03);
    convShown = Math.min(ceil, convShown + (ceil - convShown) * 0.012);
    paintConv();
  }, 120);
}
function paintConv() {
  const pct = Math.floor(Math.max(0, Math.min(1, convShown)) * 100);
  $('#conv-pct').textContent = pct;
  $('#conv-bar').style.width = pct + '%';
}

// ------------------------------------------------------------------ misc
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.toggle('err', err);
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), err ? 6000 : 2600);
}

function bindUi() {
  $('#btn-pick').onclick = () => py.openImage('');
  $('#btn-change').onclick = () => py.openImage('');
  $('#src-thumb').onclick = () => py.openImage('');
  $('#btn-example').onclick = () => S && S.example && py.openImage(S.example);
  $('#btn-open').onclick = () => py.open('');
  $('#btn-save').onclick = () => py.save('');
  const w = $('#in-width');
  const commitW = () => { const v = parseFloat(String(w.value).replace(',', '.')); if (v > 0) py.setWidth(v); };
  w.addEventListener('change', commitW);
  w.addEventListener('keydown', (e) => { if (e.key === 'Enter') { commitW(); w.blur(); } });
  $('#n-less').onclick = () => py.setColourCount(S.colours.length - 1);
  $('#n-more').onclick = () => py.setColourCount(S.colours.length + 1);
  $('#btn-recolour').onclick = () => { openCol = null; py.resetColours(); };
  $('#pop-x').onclick = () => { openCol = null; renderColours(); };
  for (const m of $$('.mode')) m.onclick = () => S.image.path && py.setMode(m.dataset.mode);
  for (const b of $$('#views button')) b.onclick = () => setView(b.dataset.view);
  $('#in-explode').addEventListener('input', (e) => { const t = parseFloat(e.target.value); if (t > 0) stage.obliqueIfFrontal(); stage.setExplode(t); });
  $('#in-panels').onchange = (e) => stage.setPanels(e.target.checked);
  $('#pl-prev').onclick = () => stepPlate(-1);
  $('#pl-next').onclick = () => stepPlate(1);
  $('#pl-all').onclick = () => { plateIdx = null; stage.focusPlate(null); renderPlateNav(); };
  $('#btn-export').onclick = () => py.export('', (r) => { if (r) { body.classList.add('exporting'); $('#export-bar').style.width = '2%'; } });
  $('#btn-cancel').onclick = () => py.cancelExport();
  $('#btn-reveal').onclick = () => { if (lastExport) py.reveal(lastExport.out_dir); };
  $('#btn-tpl-a4').onclick = () => { if (lastExport && lastExport.sablon_a4) py.reveal(lastExport.sablon_a4); };
  $('#btn-tpl-a3').onclick = () => { if (lastExport && lastExport.sablon_a3) py.reveal(lastExport.sablon_a3); };
  $('#btn-plates-dir').onclick = () => { if (lastExport) py.reveal(lastExport.plates_dir || lastExport.out_dir); };
  $('#btn-done-close').onclick = () => body.classList.remove('done-open');
  bindFine();
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === 's') { e.preventDefault(); py.save(''); }
    if (e.ctrlKey && e.key.toLowerCase() === 'o') { e.preventDefault(); py.open(''); }
    if (e.key === 'Escape') { openCol = null; renderColours(); sheet(null); stage.select(null); body.classList.remove('done-open'); }
    if (view === 'plates' && e.key === 'ArrowRight') stepPlate(1);
    if (view === 'plates' && e.key === 'ArrowLeft') stepPlate(-1);
  });
}

let lastExport = null;
function connect() {
  py.state.connect((j) => { S = JSON.parse(j); render(); });
  py.busy.connect((j) => { const b = JSON.parse(j); body.classList.toggle('busy', b.on); renderLabel(); });
  py.progress.connect((j) => converting(JSON.parse(j)));
  py.built.connect((j) => {
    built = JSON.parse(j);
    renderNumbers(); renderLabel();
    if (body.classList.contains('no3d')) loadFlat();
  });
  py.scene.connect(async (j) => {
    const s = JSON.parse(j);
    try { await stage.loadScene(s.url, s.meta); } catch (e) { py.log('scene: ' + e); }
    body.classList.toggle('draft', s.quality === 'fast');
    if (stage.selected) showPart(stage.selected.userData.meta);
  });
  py.plates.connect((j) => { if (view === 'plates') plates(JSON.parse(j)); else platesState = null; });
  py.thumb.connect((j) => { const t = JSON.parse(j); if (S) { S.thumbs[t.style] = t.url; render(); } });
  py.exportProgress.connect((f, msg) => {
    body.classList.add('exporting');
    $('#export-bar').style.width = `${Math.max(2, f * 100)}%`;
    $('#export-msg').textContent = msg || 'Hazırlanıyor';
  });
  py.exportDone.connect((j) => {
    const r = lastExport = JSON.parse(j);
    body.classList.remove('exporting');
    const bits = [];
    if (r.parts != null) bits.push(`${fmt0.format(r.parts)} taş`);
    if (r.plates != null) bits.push(`${r.plates} tabla`);
    if (r.hours != null) bits.push(`~${fmt0.format(r.hours)} saat baskı`);
    const SEP = String.fromCharCode(92);
    const segs = String(r.out_dir || '').split(/[\\/]/).filter(Boolean);
    const short = segs.length > 2 ? '…' + SEP + segs.slice(-2).join(SEP) : r.out_dir;
    const warn = (r.warnings || []).slice(0, 4).map((w) => `<li>${esc(w)}</li>`).join('');
    $('#btn-tpl-a4').style.display = r.sablon_a4 ? '' : 'none';
    $('#btn-tpl-a3').style.display = r.sablon_a3 ? '' : 'none';
    $('.done-lead').style.display = r.sablon_a4 || r.sablon_a3 ? '' : 'none';
    $('#done-s').innerHTML = `${bits.join(' · ')}<br>Bambu Studio tabla dosyaları (3MF) ve kılavuz <b title="${esc(r.out_dir)}">${esc(short)}</b> klasöründe.`
      + (warn ? `<ul class="done-w">${warn}</ul>` : '');
    body.classList.add('done-open');
  });
  py.failed.connect((j) => {
    const f = JSON.parse(j);
    if (f.kind === 'export') body.classList.remove('exporting');
    toast(f.msg, true);
  });
  py.toast.connect((m) => toast(m));
  py.plates.connect((j) => {
    S && (S.has_full = true);
    plateOf = {};
    for (const p of (JSON.parse(j).meta.plates || [])) for (const c of p.codes || []) plateOf[c] = p.title;
    if (stage.selected && stage.selected.userData.meta) showPart(stage.selected.userData.meta);
  });
}

window.tessera = {
  dragHint: (on) => body.classList.toggle('dragging', !!on),
  setView, stepPlate, sheet,
  explode: (t) => { $('#in-explode').value = t; if (t > 0) stage.obliqueIfFrontal(); stage.setExplode(t); },
  pickCode: (code) => { const o = stage.byCode && stage.byCode[code]; if (o) stage.select(o); return !!o; },
  pickAt: (fx, fy) => {
    const r = stage.renderer.domElement.getBoundingClientRect();
    stage._pick({ clientX: r.left + r.width * fx, clientY: r.top + r.height * fy });
    return stage.selected ? stage.selected.userData.meta.code : '';
  },
  ready: false,
  status: () => ({ converting: body.classList.contains('is-converting'), pct: $('#conv-pct').textContent, scene: !!stage.meta, quality: stage.meta && stage.meta.quality, plates: !!stage.platesMeta,
    busy: body.classList.contains('busy'), built: !!built, view }),
};

// ------------------------------------------------------------------ window bar (frameless window)
function bindBar() {
  for (const d of $$('[data-drag]')) {
    d.addEventListener('mousedown', (e) => { if (e.button === 0 && e.detail === 1) py.winDrag(); });
    d.addEventListener('dblclick', () => py.winToggle());
  }
  $('#w-min').onclick = () => py.winMinimize();
  $('#w-max').onclick = () => py.winToggle();
  $('#w-close').onclick = () => py.winClose();
  const edgeAt = (e) => {
    if (body.classList.contains('maximized')) return 0;
    const m = 5, x = e.clientX, y = e.clientY;
    return (y <= m ? 1 : 0) | (x <= m ? 2 : 0) | (x >= innerWidth - m ? 4 : 0) | (y >= innerHeight - m ? 8 : 0);
  };
  const cur = { 1: 'ns', 8: 'ns', 2: 'ew', 4: 'ew', 3: 'nwse', 12: 'nwse', 5: 'nesw', 10: 'nesw' };
  window.addEventListener('mousemove', (e) => {
    const k = cur[edgeAt(e)];
    document.documentElement.style.cursor = k ? k + '-resize' : '';
    body.classList.toggle('on-edge', !!k);
  }, true);
  window.addEventListener('mousedown', (e) => {
    const k = edgeAt(e);
    if (k && e.button === 0) { e.preventDefault(); e.stopPropagation(); py.winResize(k); }
  }, true);
}
window.tessera_window = (maxed) => body.classList.toggle('maximized', !!maxed);

bindUi();
bindBar();
new QWebChannel(qt.webChannelTransport, async (ch) => {
  py = ch.objects.py;
  connect();
  S = JSON.parse(await call('getState'));
  if (body.classList.contains('no3d')) { loadFlat(); toast('3D görünüm bu bilgisayarda açılamadı; düz önizleme gösteriliyor.', true); }
  render();
  const sc = await call('getSceneGLB');
  if (sc) { const s = JSON.parse(sc); stage.loadScene(s.url, s.meta); }
  window.tessera.ready = true;
});
