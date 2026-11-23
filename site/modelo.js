// Modelo de nivel en San Fernando, en el navegador. Replica explicar.armar + pronos2.marea_feats + LightGBM.
// Funciona en el navegador (window.Modelo) y en node (module.exports) para testearlo contra Python.
(function (raiz) {
  const H2000 = 262968; // horas desde 1970 hasta 2000-01-01 UTC
  const DOS_PI = 2 * Math.PI;

  function mareaNueva(m, hora) {
    const h = hora - H2000, N = DOS_PI * h / m.nodal_h, cN = Math.cos(N), sN = Math.sin(N);
    let i = 0, v = m.coef[i++];
    for (const [p, nod] of m.periodos) {
      const w = DOS_PI * h / p, c = Math.cos(w), s = Math.sin(w);
      v += m.coef[i++] * c + m.coef[i++] * s;
      if (nod) v += m.coef[i++] * c * cN + m.coef[i++] * c * sN + m.coef[i++] * s * cN + m.coef[i++] * s * sN;
    }
    return v;
  }
  function mareaVieja(m, hora) {
    const h = hora - H2000;
    let i = 0, v = m.coef[i++];
    for (const p of m.periodos) { const w = DOS_PI * h / p; v += m.coef[i++] * Math.cos(w) + m.coef[i++] * Math.sin(w); }
    return v;
  }

  // ---- series de entrada ----
  // forecast: {punto: {h0, vel[], dir[], pres[]}} (vel en m/s, dir en grados, hora = horas desde 1970 UTC)
  function serieViento(f) {
    const n = f.vel.length, u = new Float64Array(n), v = new Float64Array(n), su = new Float64Array(n), sv = new Float64Array(n), p = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const vel = f.vel[i] == null ? NaN : f.vel[i], d = f.dir[i] == null ? NaN : f.dir[i] * Math.PI / 180;
      u[i] = -vel * Math.sin(d); v[i] = -vel * Math.cos(d); su[i] = u[i] * vel; sv[i] = v[i] * vel;
      p[i] = f.pres[i] == null ? NaN : f.pres[i];
    }
    const at = (arr, h) => { const i = h - f.h0; return i >= 0 && i < n ? arr[i] : NaN; };
    const media = (arr, h, k) => { let s = 0; for (let j = 0; j < k; j++) { const x = at(arr, h - j); if (Number.isNaN(x)) return NaN; s += x; } return s / k; };
    return { u: (h) => at(u, h), v: (h) => at(v, h), p: (h) => at(p, h), su: (h, k) => media(su, h, k), sv: (h, k) => media(sv, h, k) };
  }
  // ríos (Rosario, Concordia): [[timestart ISO, valor], ...] → promedio por día UTC, corrido 1 día, ffill hasta 120 h
  function serieRio(obs) {
    const dias = new Map();
    for (const [t, val] of obs) {
      const d = Math.floor(Date.parse(t) / 864e5);
      const a = dias.get(d) || [0, 0]; a[0] += val; a[1]++; dias.set(d, a);
    }
    const pts = [...dias.entries()].map(([d, [s, n]]) => [(d + 1) * 24, s / n]).sort((a, b) => a[0] - b[0]);
    return (h) => {
      let lo = 0, hi = pts.length - 1, k = -1;
      while (lo <= hi) { const m = (lo + hi) >> 1; if (pts[m][0] <= h) { k = m; lo = m + 1; } else hi = m - 1; }
      return k >= 0 && h - pts[k][0] <= 120 ? pts[k][1] : NaN;
    };
  }

  function features(M, D, hora) {
    const f = {};
    const mv = (h) => mareaVieja(M.marea_vieja, h);
    f.marea = mv(hora);
    for (const k of [-3, -1, 1, 3]) f[`marea_${k}`] = mv(hora - k);
    const dt = new Date(hora * 3.6e6), ini = Date.UTC(dt.getUTCFullYear(), 0, 1);
    const doy = Math.floor((dt.getTime() - ini) / 864e5) + 1;
    f.doy_s = Math.sin(DOS_PI * doy / 365.25); f.doy_c = Math.cos(DOS_PI * doy / 365.25);
    for (const p of [...M.puntos_rdp, ...M.puntos_sur]) {
      const w = D.viento[p], lags = M.puntos_sur.includes(p) ? [...M.lags, 96, 120] : M.lags;
      for (const L of lags) { f[`${p}_u_${L}`] = w.u(hora - L); f[`${p}_v_${L}`] = w.v(hora - L); }
      for (const n of [6, 12, 24, 48, 72]) { f[`${p}_su_m${n}`] = w.su(hora, n); f[`${p}_sv_m${n}`] = w.sv(hora, n); }
    }
    const P = (p, h) => D.viento[p].p(h);
    for (const p of M.puntos_rdp) { f[`${p}_p`] = P(p, hora); f[`${p}_dp12`] = P(p, hora) - P(p, hora - 12); }
    for (const [a, b] of [["plataforma_sur", "delta"], ["plataforma_este", "delta"], ["boca", "delta"], ["samborombon", "rdp_medio"], ["plataforma_sur", "plataforma_este"]])
      for (const L of [0, 6, 12, 24]) f[`grad_${a}_${b}_${L}`] = P(a, hora - L) - P(b, hora - L);
    for (const p of M.puntos_sur) {
      f[`${p}_p`] = P(p, hora);
      for (const L of [0, 12, 24, 48]) f[`grad_${p}_delta_${L}`] = P(p, hora - L) - P("delta", hora - L);
    }
    const s = hora - 168; // ríos atrasados una semana (como en pronos2)
    f.con = D.con(s); f.con_d7 = D.con(s) - D.con(s - 168);
    f.ros = D.ros(s); f.ros_d7 = D.ros(s) - D.ros(s - 168); f.ros_d30 = D.ros(s) - D.ros(s - 720);
    const m2 = (h) => mareaNueva(M.marea_nueva, h);
    f.marea2 = m2(hora);
    for (const k of [-6, -4, -2, -1, 1, 2, 4, 6]) f[`marea2_${k}`] = m2(hora - k);
    for (const [c, per] of Object.entries(M.fases)) { const w = DOS_PI * (hora - H2000) / per; f[`fase_${c}_c`] = Math.cos(w); f[`fase_${c}_s`] = Math.sin(w); }
    let mx = -Infinity, mn = Infinity;
    for (let k = -12; k <= 12; k++) { const x = m2(hora + k); if (x > mx) mx = x; if (x < mn) mn = x; }
    f.marea2_rango25 = mx - mn;
    return M.cols.map((c) => (c in f ? f[c] : NaN));
  }

  function arbol(t, x) {
    let n = t.f.length ? 0 : -1;
    while (n >= 0) n = x[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    const k = -n - 1;
    if (!t.lf) return t.v[k];
    let s = t.c[k];
    const fs = t.lf[k], ks = t.lk[k];
    for (let i = 0; i < fs.length; i++) s += ks[i] * x[fs[i]];
    return s;
  }
  const bosque = (ts, x) => { let s = 0; for (const t of ts) s += arbol(t, x); return s; };
  function interp(c, x) {
    const X = c.x, Y = c.y, n = X.length;
    if (x <= X[0]) return Y[0];
    if (x >= X[n - 1]) return Y[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (X[m] <= x) lo = m; else hi = m; }
    return X[hi] === X[lo] ? Y[lo] : Y[lo] + (Y[hi] - Y[lo]) * (x - X[lo]) / (X[hi] - X[lo]);
  }

  // datos crudos → series; predice cada hora de `horas`. emision = hora de la corrida (para la antelación).
  function preparar(crudo) {
    const viento = {};
    for (const p in crudo.forecast) viento[p] = serieViento(crudo.forecast[p]);
    return { viento, ros: serieRio(crudo.rosario), con: serieRio(crudo.concordia) };
  }
  function predecir(M, D, horas, emision) {
    return horas.map((h) => {
      const x = features(M, D, h).map((v, i) => (Number.isNaN(v) ? M.medianas[i] : v));
      const reg = bosque(M.reg, x), cla = 1 / (1 + Math.exp(-bosque(M.cla, x)));
      const lead = Math.max(0, Math.min(7, Math.ceil((h - emision) / 24)));
      const c = M.cal[lead];
      return { hora: h, reg, cla, lead, nivel: interp(c.nivel, reg), prob: interp(c.prob, cla), sd: c.sd };
    });
  }

  const api = { mareaNueva, features, preparar, predecir, arbol, H2000 };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else raiz.Modelo = api;
})(typeof window !== "undefined" ? window : globalThis);
