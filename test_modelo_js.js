// Compara site/modelo.js contra Python (results/test_vector.json, generado por web_modelo.py).
// Uso: node test_modelo_js.js
const fs = require("fs");
const Modelo = require("./site/modelo.js");
const M = JSON.parse(fs.readFileSync(__dirname + "/site/model.json"));
const V = JSON.parse(fs.readFileSync(__dirname + "/results/test_vector.json"));
const D = Modelo.preparar({ forecast: V.forecast, rosario: V.rosario, concordia: V.concordia });

let peor = { d: 0 };
V.feat_horas.forEach((h, r) => {
  const js = Modelo.features(M, D, h);
  V.features[r].forEach((py, i) => {
    const a = py == null ? NaN : py, b = js[i];
    if (Number.isNaN(a) !== Number.isNaN(b)) { peor = { d: Infinity, col: M.cols[i], h, py: a, js: b }; return; }
    const d = Number.isNaN(a) ? 0 : Math.abs(a - b);
    if (d > peor.d) peor = { d, col: M.cols[i], h, py: a, js: b };
  });
});
console.log("features: mayor diferencia", peor);

const pr = Modelo.predecir(M, D, V.horas, V.ahora);
let dr = 0, dc = 0;
pr.forEach((p, i) => { dr = Math.max(dr, Math.abs(p.reg - V.reg[i])); dc = Math.max(dc, Math.abs(p.cla - V.cla[i])); });
console.log(`predicciones (${pr.length} horas): |Δ nivel| máx ${dr.toExponential(2)} m, |Δ prob| máx ${dc.toExponential(2)}`);
const ok = peor.d < 1e-6 && dr < 1e-6 && dc < 1e-6;
console.log(ok ? "OK: JS = Python" : "DIFERENCIAS");
process.exit(ok ? 0 : 1);
