// Archiva en el Sheet: observaciones INA (San Fernando, Tigre), pronóstico INA de San Fernando y pronóstico SHN.
// Llamar update_ina() desde el mismo trigger que update_google_sheet(). Se puede correr seguido: no duplica filas.

var INA = "https://alerta.ina.gob.ar/a5";
var TZ = "America/Argentina/Buenos_Aires";
var INA_ESTACIONES = {52: "san fernando", 49: "tigre"};

function hoja(nombre, encabezado) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(nombre);
  if (!sh) {
    sh = ss.insertSheet(nombre);
    sh.appendRow(encabezado);
  }
  return sh;
}

function agregar(sh, filas) {
  if (filas.length) sh.getRange(sh.getLastRow() + 1, 1, filas.length, filas[0].length).setValues(filas);
}

function fmt(d) {
  return Utilities.formatDate(d, TZ, "yyyy-MM-dd HH:mm:ss");
}

function getJson(url) {
  return JSON.parse(UrlFetchApp.fetch(url).getContentText());
}

function prop(k, v) {
  var p = PropertiesService.getScriptProperties();
  if (v === undefined) return p.getProperty(k);
  p.setProperty(k, String(v));
}

function ina_obs() {
  var sh = hoja("ina-obs", ["fecha", "estacion", "altura_m"]);
  var hasta = new Date(Date.now() + 864e5).toISOString().slice(0, 10);
  var desde = new Date(Date.now() - 3 * 864e5).toISOString().slice(0, 10);
  for (var id in INA_ESTACIONES) {
    var ultimo = prop("ina_obs_" + id) || "";
    var obs = getJson(INA + "/obs/puntual/series/" + id + "/observaciones?timestart=" + desde + "&timeend=" + hasta + "&format=json");
    obs.sort(function (a, b) { return a.timestart < b.timestart ? -1 : 1; });
    var filas = obs.filter(function (o) { return o.timestart > ultimo; })
      .map(function (o) { return [fmt(new Date(o.timestart)), INA_ESTACIONES[id], o.valor]; });
    agregar(sh, filas);
    if (obs.length) prop("ina_obs_" + id, obs[obs.length - 1].timestart);
  }
}

function ina_prono() {
  var sh = hoja("ina-prono", ["bajado", "cor_id", "emitido", "fecha", "main", "p05", "p25", "p75", "p95"]);
  var d = getJson(INA + "/sim/calibrados/432/corridas/last?series_id=26202&includeProno=true&format=json");
  if (String(d.id) === prop("ina_prono_cor")) return;
  var emitido = new Date(d.forecast_date);
  var porHora = {};
  d.series[0].pronosticos.forEach(function (p) {
    if (new Date(p.timestart) < emitido) return;  // la parte previa a la emisión es el ajuste del modelo, no pronóstico
    (porHora[p.timestart] = porHora[p.timestart] || {})[p.qualifier || "main"] = p.valor;
  });
  var ahora = fmt(new Date());
  var filas = Object.keys(porHora).sort().map(function (t) {
    var q = porHora[t];
    return [ahora, d.id, fmt(emitido), fmt(new Date(t)), q.main, q.p05, q.p25, q.p75, q.p95];
  });
  agregar(sh, filas);
  prop("ina_prono_cor", d.id);
}

function shn_prono() {
  var sh = hoja("shn-prono", ["bajado", "lugar", "estado", "fecha", "altura_m", "aviso"]);
  var html = UrlFetchApp.fetch("https://www.hidro.gov.ar/oceanografia/pronostico.asp").getContentText();
  var lineas = html.replace(/<[^>]+>/g, "\n").replace(/&nbsp;/g, " ").split("\n")
    .map(function (l) { return l.trim(); }).filter(String);
  var ahora = fmt(new Date()), aviso = "", lugar = null, filas = [];
  for (var i = 0; i < lineas.length; i++) {
    var l = lineas[i];
    if (l.indexOf("Corrección a las alturas") === 0 || l.indexOf("Correcci&oacute;n a las alturas") === 0) {
      aviso = lineas[i + 1];
    } else if (l === "PLEAMAR" || l === "BAJAMAR") {
      if (!lugar || i + 3 >= lineas.length) continue;
      var f = lineas[i + 3].split("/");
      filas.push([ahora, lugar, l, f[2] + "-" + f[1] + "-" + f[0] + " " + lineas[i + 1] + ":00", parseFloat(lineas[i + 2]), aviso]);
    } else if ((lineas[i + 1] === "PLEAMAR" || lineas[i + 1] === "BAJAMAR") && !/^\d+\/\d+\/\d+$/.test(l)) {
      lugar = l;
    }
  }
  var hash = Utilities.base64Encode(Utilities.computeDigest(Utilities.DigestAlgorithm.MD5, JSON.stringify(filas.map(function (r) { return r.slice(1); }))));
  if (hash === prop("shn_hash")) return;
  agregar(sh, filas);
  prop("shn_hash", hash);
}

function update_ina() {
  [ina_obs, ina_prono, shn_prono].forEach(function (f) {
    try { f(); } catch (e) { console.error(f.name + ": " + e); }
  });
}
