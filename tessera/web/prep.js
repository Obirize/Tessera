// "Görseli hazırla": crop, outline, size and piece count of a picture, with its real stones beside it.
import { fmt0, pieceControl, plateOffer } from './pieces.js';

const $ = (s) => document.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const RATIOS = { kare: 1, dikey: 3 / 4, yatay: 4 / 3 };   // width / height

export function makePrep() {
  let prep = null, P = null, R = null, Z = null, tab = 'gorsel', busy = false, ratio = 'tamam', err = '';
  const send = (d) => prep.set(JSON.stringify(d));
  const root = $('#prep'), view = $('#p-view'), img = $('#p-img'), svg = $('#p-svg');

  // ---------------------------------------------------------------- crop box on the picture
  let box = [0, 0, 1, 1];                       // what is drawn; P.kutu is what Python has
  let rect = { x: 0, y: 0, w: 1, h: 1 };        // where the picture lies in the view, px
  const round = () => P && P.sekil.tur === 'daire';
  const lock = () => (round() ? 1 : RATIOS[ratio] || null);

  function place() {
    if (!P) return;
    const pad = 22, W = view.clientWidth - 2 * pad, H = view.clientHeight - 2 * pad;
    const k = Math.min(W / P.w, H / P.h);
    rect = { w: P.w * k, h: P.h * k };
    rect.x = (view.clientWidth - rect.w) / 2;
    rect.y = (view.clientHeight - rect.h) / 2;
    Object.assign(img.style, { left: rect.x + 'px', top: rect.y + 'px', width: rect.w + 'px', height: rect.h + 'px' });
    draw();
  }
  function shapePath(x, y, w, h) {
    const t = P.sekil.tur;
    if (t === 'daire' || t === 'oval') {
      const a = w / 2, b = h / 2;
      return `M${x} ${y + b}a${a} ${b} 0 1 0 ${w} 0a${a} ${b} 0 1 0 ${-w} 0z`;
    }
    const r = t === 'yuvarlak' ? Math.min(0.499, P.sekil.r) * Math.min(w, h) : 0;
    if (!r) return `M${x} ${y}h${w}v${h}h${-w}z`;
    return `M${x + r} ${y}h${w - 2 * r}a${r} ${r} 0 0 1 ${r} ${r}v${h - 2 * r}a${r} ${r} 0 0 1 ${-r} ${r}h${-(w - 2 * r)}a${r} ${r} 0 0 1 ${-r} ${-r}v${-(h - 2 * r)}a${r} ${r} 0 0 1 ${r} ${-r}z`;
  }
  const el = (n, a, p) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); (p || svg).appendChild(e); return e; };
  function draw() {
    svg.innerHTML = '';
    if (!P) return;
    const [x0, y0, x1, y1] = box;
    const x = rect.x + x0 * rect.w, y = rect.y + y0 * rect.h, w = (x1 - x0) * rect.w, h = (y1 - y0) * rect.h;
    const sh = shapePath(x, y, w, h);
    el('path', { d: `M${rect.x} ${rect.y}h${rect.w}v${rect.h}h${-rect.w}z${sh}`, class: 'p-dark', 'fill-rule': 'evenodd' });
    const same = (a, b) => a && b && a.every((v, i) => Math.abs(v - b[i]) < 1e-4);
    if (P.zemin && Z && Z.url && same(Z.kutu || [0, 0, 1, 1], box)) {
      el('image', { href: Z.url, x, y, width: w, height: h, preserveAspectRatio: 'none', class: 'p-cut' });
    }
    el('rect', { x, y, width: w, height: h, class: 'p-box', 'data-h': 'm' });
    if (P.sekil.tur !== 'dikdortgen') el('path', { d: sh, class: 'p-shape' });
    for (const t of [1 / 3, 2 / 3]) {
      el('path', { d: `M${x + w * t} ${y}v${h}M${x} ${y + h * t}h${w}`, class: 'p-third' });
    }
    const pts = { nw: [0, 0], n: [.5, 0], ne: [1, 0], e: [1, .5], se: [1, 1], s: [.5, 1], sw: [0, 1], w: [0, .5] };
    for (const k in pts) {
      const [u, v] = pts[k], cx = x + u * w, cy = y + v * h;
      el('rect', { x: cx - 5, y: cy - 5, width: 10, height: 10, class: 'p-knob' });
      el('rect', { x: cx - 14, y: cy - 14, width: 28, height: 28, class: 'p-grab', 'data-h': k, style: `cursor:${k}-resize` });
    }
  }
  // box of the wanted ratio (picture pixels), around a centre, inside the picture
  function fitted(q, cx, cy, bw) {
    let w = bw, h = bw * P.w / q / P.h;
    const k = Math.min(1, 1 / w, 1 / h);
    w *= k; h *= k;
    const x0 = Math.min(1 - w, Math.max(0, cx - w / 2)), y0 = Math.min(1 - h, Math.max(0, cy - h / 2));
    return [x0, y0, x0 + w, y0 + h];
  }
  let dragging = null;
  svg.addEventListener('pointerdown', (e) => {
    const h = e.target.dataset && e.target.dataset.h;
    if (!h || !P) return;
    dragging = { h, x: e.clientX, y: e.clientY, b: [...box], shift: e.shiftKey, moved: false };
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const { h, b } = dragging;
    if (!dragging.moved && Math.hypot(e.clientX - dragging.x, e.clientY - dragging.y) < 4) return;
    dragging.moved = true;
    const dx = (e.clientX - dragging.x) / rect.w, dy = (e.clientY - dragging.y) / rect.h;
    let [x0, y0, x1, y1] = b;
    if (h === 'm') {
      const w = x1 - x0, t = y1 - y0;
      x0 = Math.min(1 - w, Math.max(0, x0 + dx)); y0 = Math.min(1 - t, Math.max(0, y0 + dy));
      box = [x0, y0, x0 + w, y0 + t];
    } else {
      const m = 0.05;
      if (h.includes('w')) x0 = Math.max(0, Math.min(x1 - m, x0 + dx));
      if (h.includes('e')) x1 = Math.min(1, Math.max(x0 + m, x1 + dx));
      if (h.includes('n')) y0 = Math.max(0, Math.min(y1 - m, y0 + dy));
      if (h.includes('s')) y1 = Math.min(1, Math.max(y0 + m, y1 + dy));
      const q = lock();
      if (q) {                                   // keep the ratio; the side or corner across stays where it is
        const fx = h.includes('w') ? b[2] : h.includes('e') ? b[0] : (b[0] + b[2]) / 2;
        const fy = h.includes('n') ? b[3] : h.includes('s') ? b[1] : (b[1] + b[3]) / 2;
        const sx = h.includes('w') ? -1 : h.includes('e') ? 1 : 0, sy = h.includes('n') ? -1 : h.includes('s') ? 1 : 0;
        const roomX = sx < 0 ? fx : sx > 0 ? 1 - fx : 2 * Math.min(fx, 1 - fx);
        const roomY = sy < 0 ? fy : sy > 0 ? 1 - fy : 2 * Math.min(fy, 1 - fy);
        let w = sx ? x1 - x0 : (y1 - y0) * P.h * q / P.w;
        if (sx && sy) w = Math.max(w, (y1 - y0) * P.h * q / P.w);
        w = Math.min(w, roomX, roomY * P.h * q / P.w);
        const t = w * P.w / q / P.h;
        x0 = sx < 0 ? fx - w : sx > 0 ? fx : fx - w / 2; x1 = x0 + w;
        y0 = sy < 0 ? fy - t : sy > 0 ? fy : fy - t / 2; y1 = y0 + t;
      } else if (ratio === 'tamam') { ratio = 'serbest'; renderRatio(); }
      box = [x0, y0, x1, y1];
    }
    draw();
  });
  const drop = () => {
    if (!dragging) return;
    const d = dragging;
    dragging = null;
    if (!d.moved && d.h === 'm' && P.zemin) {    // a click, not a drag: this area goes or comes back
      const r = svg.getBoundingClientRect(), [x0, y0, x1, y1] = box;
      const fx = (d.x - r.left - rect.x - x0 * rect.w) / ((x1 - x0) * rect.w);
      const fy = (d.y - r.top - rect.y - y0 * rect.h) / ((y1 - y0) * rect.h);
      if (fx >= 0 && fx <= 1 && fy >= 0 && fy <= 1) send({ tik: [fx, fy, d.shift ? 'renk' : 'tek'] });
      return;
    }
    if (box.some((v, i) => Math.abs(v - P.kutu[i]) > 1e-4)) send({ kutu: box });
  };
  svg.addEventListener('pointerup', drop);
  svg.addEventListener('pointercancel', drop);
  new ResizeObserver(place).observe(view);

  function setRatio(v) {
    ratio = v;
    if (v === 'tamam') box = [0, 0, 1, 1];
    else if (RATIOS[v]) box = fitted(RATIOS[v], (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, Math.max(box[2] - box[0], (box[3] - box[1]) * P.h * RATIOS[v] / P.w));
    renderRatio(); draw();
    if (v !== 'serbest') send({ kutu: box });
  }
  function renderRatio() {
    for (const b of $('#p-ratio').children) {
      b.classList.toggle('on', round() ? b.dataset.v === 'kare' : b.dataset.v === ratio);
      b.disabled = round() && b.dataset.v !== 'kare';
    }
  }

  // ---------------------------------------------------------------- the rest of the window
  const piece = pieceControl({ box: $('#p-piece'), seg: $('#p-pc-mode'), num: $('#p-pc'), range: $('#p-pc-r'), note: $('#p-pc-s') },
    () => ({ want: P && P.parca, p: R && R.parca, size: R && R.size_mm }), (v) => send({ parca: v }));

  function render() {
    root.classList.toggle('on', !!P);
    document.body.classList.toggle('prep-open', !!P);
    if (!P) return;
    root.classList.toggle('taslar', tab === 'taslar');
    root.classList.toggle('busy', busy);
    for (const b of $('#p-tab').children) b.classList.toggle('on', b.dataset.v === tab);
    $('#p-ad').textContent = P.ad;
    $('#p-ok').textContent = P.yeni ? 'Ekle' : 'Uygula';
    $('#p-other').style.visibility = P.yeni ? 'visible' : 'hidden';
    if (img.getAttribute('src') !== P.src) img.src = P.src;
    for (const b of $('#p-shape').children) b.classList.toggle('on', b.dataset.v === P.sekil.tur);
    $('#p-rr').classList.toggle('on', P.sekil.tur === 'yuvarlak');
    if (P.sekil.r && document.activeElement !== $('#p-r')) $('#p-r').value = P.sekil.r;
    const w = $('#p-w');
    if (document.activeElement !== w) w.value = fmt0.format(P.width_cm).replace(/\./g, '');
    const cut = [(P.kutu[2] - P.kutu[0]) * P.w, (P.kutu[3] - P.kutu[1]) * P.h];
    $('#p-h').textContent = `× ${fmt0.format(P.width_cm * cut[1] / cut[0])} cm`;
    renderRatio();
    root.classList.toggle('zemin', !!P.zemin);
    for (const b of $('#p-bg').children) b.classList.toggle('on', (b.dataset.v === 'kaldir') === !!P.zemin);
    const zs = P.zemin && Z && Z.soz;
    $('#p-bg-fit').className = 'fit on ' + (zs ? Z.puan : 'orta');
    $('#p-bg-t').innerHTML = zs ? Z.soz.map((t) => `<p>${t}</p>`).join('') : '<p>Arka plan aranıyor…</p>';
    $('#p-bg-geri').style.visibility = P.zemin && P.zemin.tiklar.length ? 'visible' : 'hidden';
    piece(busy);
    plateOffer({ box: $('#p-base'), text: $('#p-base-t'), check: $('#p-base-c') }, P.tek_taban, R && R.parca, P.hexes, (v) => send({ tek_taban: v }));
    $('#p-fit').className = 'fit' + (R && R.soz && R.soz.length && !busy ? ' on ' + R.puan : '');
    $('#p-fit-t').textContent = R && R.soz ? R.soz.join(' ') : '';
    const st = $('#p-stones');
    if (R && R.url && st.getAttribute('src') !== R.url) st.src = R.url;
    st.style.visibility = R && R.url ? 'visible' : 'hidden';
    const t = R && R.tahmin;
    $('#p-num').innerHTML = err ? `<span class="p-err">${err}</span>` : R && R.parca && R.parca.sayi && !busy
      ? `<span><b>${fmt0.format(R.parca.sayi)}</b>taş</span><span><b>${fmt0.format(R.parca.orta_mm)} mm</b>ortalama taş</span>`
        + (t && t.tabla ? `<span><b>${t.tabla}</b>${P.mode === 'ams' ? 'panel' : 'tabla'}</span><span><b>~${fmt0.format(t.saat)}</b>saat</span>` : '')
      : '<span>Taşlar hesaplanıyor…</span>';
  }

  // ---------------------------------------------------------------- wiring
  $('#p-tab').onclick = (e) => { const b = e.target.closest('button'); if (b) { tab = b.dataset.v; render(); place(); } };
  $('#p-ratio').onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) setRatio(b.dataset.v); };
  $('#p-don').onclick = () => { if (RATIOS[ratio] && ratio !== 'kare') ratio = ratio === 'dikey' ? 'yatay' : 'dikey'; send({ op: 'don' }); };
  $('#p-ayna').onclick = () => send({ op: 'ayna' });
  $('#p-bg').onclick = (e) => { const b = e.target.closest('button'); if (b) { Z = null; send({ zemin: b.dataset.v === 'kaldir' }); } };
  $('#p-bg-geri').onclick = () => send({ geri: true });
  $('#p-shape').onclick = (e) => { const b = e.target.closest('button'); if (b) send({ sekil: { tur: b.dataset.v, r: parseFloat($('#p-r').value) } }); };
  $('#p-r').oninput = () => { if (P) { P.sekil.r = parseFloat($('#p-r').value); draw(); } };
  $('#p-r').onchange = () => send({ sekil: { tur: 'yuvarlak', r: parseFloat($('#p-r').value) } });
  const w = $('#p-w');
  const commitW = () => { const v = parseFloat(String(w.value).replace(',', '.')); if (v > 0) send({ width_cm: v }); };
  w.addEventListener('change', commitW);
  w.addEventListener('keydown', (e) => { if (e.key === 'Enter') { commitW(); w.blur(); } });
  $('#p-ok').onclick = () => prep.accept();
  $('#p-cancel').onclick = () => prep.cancel();
  $('#p-other').onclick = () => prep.open('', () => {});

  return {
    connect(obj) {
      prep = obj;
      prep.state.connect((j) => {
        const s = JSON.parse(j), was = P;
        P = s.open ? s : null;
        if (!P) { R = null; Z = null; err = ''; busy = false; render(); return; }
        if (!was || was.ad !== P.ad) { R = null; Z = null; tab = 'gorsel'; ratio = P.kutu.every((v, i) => v === [0, 0, 1, 1][i]) ? 'tamam' : 'serbest'; }
        box = [...P.kutu];
        render();
        place();
      });
      prep.result.connect((j) => {
        const d = JSON.parse(j);
        if (d.zemin) { Z = d.zemin; render(); draw(); return; }   // the stones are still coming
        R = { ...(R || {}), ...d }; busy = false; err = ''; render();
      });
      prep.progress.connect((j) => {
        const d = JSON.parse(j);
        if (d.err) { err = d.err; busy = false; render(); return; }
        if (!busy || d.f === 0) { busy = true; err = ''; render(); }
        $('#p-bar').style.width = `${Math.round(100 * d.f)}%`;
        $('#p-msg').textContent = d.msg || '';
      });
    },
    get open() { return !!P; },
    start: (path) => prep.open(path || '', () => {}),
    edit: () => prep.edit(() => {}),
    cancel: () => prep.cancel(),
  };
}
