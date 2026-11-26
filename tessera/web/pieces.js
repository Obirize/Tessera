// How many stones: what the panel and the preparation window share.
const nf = new Intl.NumberFormat('tr-TR', { maximumFractionDigits: 0 });
export const fmt0 = nf;

// the slider runs from the fewest to the most sensible number; equal steps are equal ratios
export const pcAt = (v, p) => Math.round(p.en_az * Math.pow(p.en_cok / p.en_az, v / 1000));
export const pcPos = (n, p) => Math.max(0, Math.min(1000, Math.round(1000 * Math.log(n / p.en_az) / Math.log(p.en_cok / p.en_az))));
export const pcMode = (want) => (want == null ? 'auto' : want === 'sekil' ? 'sekil' : 'sayi');

// where a number starts: about 2 cm stones, every shape kept
export function pcFirst(p, sizeMm) {
  if (!p || !p.alan || !sizeMm) return 100;
  return Math.round(Math.max(p.alan, Math.min(p.en_cok, sizeMm[0] * sizeMm[1] / 400)));
}

export function pcNote(p, want) {
  const mode = pcMode(want), n = p.sayi;
  const link = (v, t) => `<button class="link small" data-pc="${v}">${t}</button>`;
  if (mode === 'sekil') {
    return `${nf.format(n)} taş. Hiçbir şekil kaybolmaz` + (n > p.alan ? `; ${nf.format(n - p.alan)} yerde uzun şekiller bölündü.` : '.');
  }
  if (mode === 'auto') return `${nf.format(n)} taş. Görselde ${nf.format(p.alan)} ayrı şekil var.`;
  const out = [];
  out.push(n === want ? `${nf.format(n)} taş, ortalama ${nf.format(p.orta_mm)} mm.` : `${nf.format(n)} taş çıktı (istediğin ${nf.format(want)}).`);
  if (want < p.alan && n < p.alan) out.push(`${nf.format(p.alan - n)} küçük şekil komşusuna katıldı. ${link('sekil', 'Her şekil kalsın')}`);
  else if (p.dar > 0.15 * n && p.rahat < 0.8 * n) out.push(`${nf.format(p.dar)} tanesi 2,5 mm'den dar, dizmesi zor. ${link(p.rahat, `${nf.format(p.rahat)} taş yap`)}`);
  return out.join(' ');
}

// the control itself: seg buttons, number, slider, note. Returns render(state).
// el: {box, seg, num, range, note}; get(): {want, p, size}; set(value)
export function pieceControl(el, get, set) {
  let drag = false;
  const commit = () => { const v = parseInt(el.num.value, 10); if (v >= 2 && v !== get().want) set(Math.min(20000, v)); };
  el.seg.onclick = (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    const g = get();
    set(b.dataset.v === 'auto' ? null : b.dataset.v === 'sekil' ? 'sekil' : typeof g.want === 'number' ? g.want : pcFirst(g.p, g.size));
  };
  el.num.addEventListener('change', commit);
  el.num.addEventListener('keydown', (e) => { if (e.key === 'Enter') { commit(); el.num.blur(); } });
  el.range.addEventListener('input', () => { drag = true; const p = get().p; if (p && p.alan) el.num.value = pcAt(+el.range.value, p); });
  el.range.addEventListener('change', () => { drag = false; commit(); });
  el.note.addEventListener('click', (e) => { const b = e.target.closest('[data-pc]'); if (b) set(b.dataset.pc === 'sekil' ? 'sekil' : +b.dataset.pc); });
  return (busy) => {
    const { want, p } = get(), ok = p && p.alan;
    el.box.classList.toggle('sayi', pcMode(want) === 'sayi');
    for (const b of el.seg.children) b.classList.toggle('on', b.dataset.v === pcMode(want));
    if (pcMode(want) === 'sayi' && !drag) {
      if (document.activeElement !== el.num) el.num.value = want;
      if (ok) el.range.value = pcPos(want, p);
    }
    el.note.innerHTML = ok && !busy ? pcNote(p, want) : '';
  };
}

// the one-piece plate: offered when one colour draws the lines, or already chosen
export function plateOffer(el, own, p, hexes, set) {
  const c = p && p.cizgi, hex = own || (c && c.hex);
  el.box.classList.toggle('none', !hex);
  if (!hex) return;
  const name = Object.keys(hexes).find((n) => hexes[n].toUpperCase() === hex.toUpperCase()) || (c && c.ad) || '';
  el.text.textContent = `${name} tek parça taban olsun, taşlar içine otursun`;
  el.check.checked = !!own;
  el.check.onchange = (e) => set(e.target.checked ? hex.toUpperCase() : null);
}
