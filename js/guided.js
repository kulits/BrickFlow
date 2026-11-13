const ICON = (d) => `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="${d}" fill="currentColor"/></svg>`;
const PLAY = ICON('M4 2.5v11l9-5.5z') + 'Guide', PAUSE = ICON('M4 2.5h3v11H4zM9 2.5h3v11H9z') + 'Pause';
const fmtE = (E) => (E < 1e-10 ? '0' : E < 0.01 ? E.toExponential(1).replace(/e(-?\d+)/, '×10<sup>$1</sup>') : E.toFixed(E < 100 ? 2 : 0));

export async function guided(el, { payload, frame, speed, status, already, onEval, onPath, onGuide }) {
  const $ = (q) => el.querySelector(q), guide = $('[data-act="guide"]'), slider = $('[data-s]'), chip = $('[data-chip]');
  const atlas = await (await fetch('data/duck_atlas.json')).json();
  const worker = new Worker(new URL('./energy_worker.js', import.meta.url), { type: 'module' });
  let ready = false, inflight = false, dirty = false, id = 0, path = [], at = 0, playing = false, held = false, last = 0, raf = 0;

  function showE(st) {
    const ok = st.E < 1e-10;
    $('[data-e]').innerHTML = fmtE(st.E);
    chip.textContent = status(st, ok);
    chip.classList.toggle('ok', ok);
  }
  const total = () => (path.length ? path[path.length - 1].tau : 0);
  function show(s) {
    at = s;
    let i = 0;
    while (i < path.length - 1 && path[i + 1].tau <= s) i++;
    const a = path[i], b = path[Math.min(i + 1, path.length - 1)], f = b.tau > a.tau ? Math.min(1, Math.max(0, (s - a.tau) / (b.tau - a.tau))) : 0;
    onPath(a, f, b);
    showE(f < 0.5 ? a : b);
    slider.value = total() > 0 ? Math.round(at / total() * 1000) : 0;
  }
  function play(on) {
    if (on !== playing) guide.innerHTML = on ? PAUSE : PLAY;
    playing = on;
    last = 0;
    if (on && !raf) raf = requestAnimationFrame(tick);
  }
  function tick(ts) {
    raf = 0;
    if (!playing) return;
    const dt = Math.min(0.25, (ts - (last || ts)) / 1000);
    last = ts;
    show(Math.min(total(), at + Math.max(speed, total() / 3) * dt));
    if (at >= total()) play(false);
    else raf = requestAnimationFrame(tick);
  }
  function clear() {
    path = []; at = 0; held = false; id++;
    play(false);
    slider.value = 0; slider.disabled = true;
    guide.classList.remove('on');
  }
  function evaluate() {
    if (!ready || path.length) return;
    dirty = inflight;
    if (inflight) return;
    inflight = true;
    worker.postMessage({ parts: payload() });
  }

  worker.onmessage = ({ data: m }) => {
    if (m.ready) { ready = true; return evaluate(); }
    if (m.states) {
      if (m.flow !== id) return;
      guide.classList.remove('on');
      if (m.states.length > 1) { path = m.states; slider.disabled = false; if (!held) play(true); return; }
      showE(m.states[0]);
      if (m.states[0].E < 1e-10) chip.textContent = already;
      chip.classList.remove('flash'); void chip.offsetWidth; chip.classList.add('flash');
      return;
    }
    inflight = false;
    onEval(m);
    showE(m);
    if (dirty) evaluate();
  };
  worker.postMessage({ atlas });

  slider.addEventListener('input', () => { held = true; play(false); show(slider.value / 1000 * total()); });
  guide.addEventListener('click', () => {
    if (path.length > 1) {
      if (playing) { held = true; play(false); } else { held = false; if (at >= total()) show(0); play(true); }
      return;
    }
    onGuide();
    clear();
    guide.classList.add('on');
    worker.postMessage({ flow: id, parts: payload(), frame });
  });
  clear();
  return { evaluate, clear, com: (name) => atlas.parts[(atlas.alias || {})[name] || name].moments.slice(1, 4) };
}
