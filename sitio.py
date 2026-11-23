"""Genera web/lancha/data.json para el sitio: marea (coeficientes), nivel medido, pronóstico de hoy y tabla de riesgo.

1. Entrena (una vez, cacheado en results/vivo.pkl) el modelo de pronos2.py: LightGBM linear_tree + clasificador,
   con calibración isotónica por antelación (0..7 días) usando pronósticos ECMWF archivados 2024-02 → 2026-11.
2. Baja el pronóstico ECMWF de hoy (Open-Meteo) para los 10 puntos y pronostica las próximas ~8 días.
3. Tabla de riesgo histórico (marea mínima del tramo × El Niño) y factor por Rosario, como calendario.py.
Uso: .venv/bin/python -u -W ignore sitio.py   (re-correr solo el paso 2-3 es rápido una vez entrenado)
"""
import json
import os
import pickle
import socket
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

import calendario as C
import empujar as P
import explicar as E
import modelo as M
import pronos as PR
import pronos2 as PR2

ROOT = Path(__file__).parent
OUT = ROOT / "web/lancha"
CACHE = ROOT / "results/vivo.pkl"
PROG = Path.home() / ".claude/progress/sitio.json"
PUNTOS = E.PUNTOS + E.SUR
U = 0.4
ROS_COLS = ["ros", "ros_d7", "ros_d30", "con", "con_d7"]
FERIADOS = ["2026-12-08", "2026-12-25", "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-24", "2027-03-26",
            "2027-04-02", "2027-05-01", "2027-05-25", "2027-06-20", "2027-07-09", "2027-12-08", "2027-12-25"]


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "sitio", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def entrenar():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    F0, g = E.armar()
    o, _ = M.cargar()
    for c in ROS_COLS:
        F0[c] = F0[c].shift(24 * 7)
    tide = PR2.marea_feats(F0, o)
    cols = [c for k, v in g.items() if k != "rio_24h_antes" for c in v] + tide
    F0 = F0.dropna(subset=["y", "marea"])
    idx = F0.index
    S = PR.fuentes(idx)
    tr = F0[idx < PR.CORTE]
    calm = idx >= PR.CORTE
    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, linear_tree=True, linear_lambda=1.0, verbose=-1)
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    Xtr = imp.transform(tr[cols])
    reg = lgb.LGBMRegressor(**params).fit(Xtr, tr.y); progreso(1, 11)
    cla = lgb.LGBMClassifier(**{**params, "linear_tree": False, "num_leaves": 63}).fit(Xtr, (tr.y < U).astype(int)); progreso(2, 11)
    y = F0.y[calm]
    cal = {}
    for d in range(8):
        F = F0 if d == 0 else PR.armar_lead(F0, S, d, idx)
        X = imp.transform(F.loc[calm, cols])
        p, q = reg.predict(X), cla.predict_proba(X)[:, 1]
        cal[d] = (IsotonicRegression(out_of_bounds="clip").fit(p, y),
                  IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q, (y < U).astype(float)),
                  float(np.std(y - IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p))))
        progreso(3 + d, 11)
    m = dict(imp=imp, reg=reg, cla=cla, cal=cal, cols=cols)
    pickle.dump(m, open(CACHE, "wb"))
    return m


def bajar_pronostico():
    fc = {}
    for p in PUNTOS:
        lat, lon = pd.read_csv(ROOT / f"data/era5/{p}.csv", nrows=1)[["latitude", "longitude"]].iloc[0]
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               "&hourly=wind_speed_10m,wind_direction_10m,pressure_msl&models=ecmwf_ifs025"
               "&past_days=7&forecast_days=8&wind_speed_unit=ms&timezone=GMT")
        h = json.load(urllib.request.urlopen(url, timeout=120))["hourly"]
        d = pd.DataFrame(h)
        d["time"] = pd.to_datetime(d.time)
        d = d.set_index("time")
        rad = np.deg2rad(d.wind_direction_10m)
        fc[p] = pd.DataFrame({"u": -d.wind_speed_10m * np.sin(rad), "v": -d.wind_speed_10m * np.cos(rad),
                              "vel": d.wind_speed_10m, "pres": d.pressure_msl, "dir": d.wind_direction_10m}).dropna()
        time.sleep(1)
    return fc


def pronosticar(m, fc):
    ahora = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    fin = min(f.index.max() for f in fc.values())
    era5_orig, cargar_orig = E.era5, M.cargar

    def era5_mix(p, idx):
        return fc[p][["u", "v", "vel", "pres"]].reindex(idx).combine_first(era5_orig(p, idx))

    def cargar_ext():
        o, w = cargar_orig()
        idx = pd.date_range(o.index[0], fin, freq="h")
        return o.reindex(idx), w.reindex(idx).ffill(limit=24 * 10)

    E.era5, M.cargar = era5_mix, cargar_ext
    try:
        F, g = E.armar()
        o, _ = M.cargar()
    finally:
        E.era5, M.cargar = era5_orig, cargar_orig
    for c in ROS_COLS:
        F[c] = F[c].shift(24 * 7)
    PR2.marea_feats(F, o)
    F = F[F.index >= ahora - pd.Timedelta(days=3)]
    X = m["imp"].transform(F[m["cols"]])
    p, q = m["reg"].predict(X), m["cla"].predict_proba(X)[:, 1]
    lead = np.clip(np.ceil((F.index - ahora) / pd.Timedelta(hours=24)), 0, 7).astype(int)
    nivel = np.array([m["cal"][d][0].predict([x])[0] for d, x in zip(lead, p)])
    prob = np.array([m["cal"][d][1].predict([x])[0] for d, x in zip(lead, q)])
    sd = np.array([m["cal"][d][2] for d in lead])
    return F.index, nivel, prob, sd, ahora


def riesgo_historico(coef):
    o, w = M.cargar()
    idx = pd.date_range(o.index[0], o.index[-1] + pd.Timedelta(days=2), freq="h")
    marea = pd.Series(P.base_marea(idx) @ coef, index=idx)
    oni = C.oni_serie()
    dias = pd.date_range("2011-01-02", o.index[-1] - pd.Timedelta(days=1), freq="D").date
    h = C.viajes(dias)
    h["marea_min"] = [marea.reindex(C.ventana_horas(s)).min() for s in h.salida]
    h["obs_min"] = [o.reindex(C.ventana_horas(s)).min() for s in h.salida]
    h = h.dropna(subset=["obs_min"])
    h["nino"] = oni.reindex(pd.DatetimeIndex(h.salida).to_period("M").to_timestamp()).values > 0.5
    h["bin"] = pd.cut(h.marea_min, C.BINS, labels=False)
    tabla = {}
    for thr in C.UMBRALES:
        t = h.assign(b=h.obs_min < thr).groupby(["nino", "bin"]).b.mean()
        tabla[str(thr)] = {("nino" if n else "normal"): [None if np.isnan(x := float(t.get((n, k), np.nan))) else round(x, 4) for k in range(len(C.BINS) - 1)]
                           for n in [False, True]}
    ros = w.ros.dropna()
    ros_hoy = float(ros.iloc[-1])
    estado = "alto" if ros_hoy > C.ROS_ALTO else ("bajo" if ros_hoy < C.ROS_BAJO else "normal")
    hn = h[h.nino].copy()
    sd_ = pd.DatetimeIndex(hn.salida).floor("D")
    ajuste = {}
    for thr in C.UMBRALES:
        bajo = (hn.obs_min < thr).values
        ajuste[str(thr)] = []
        for a, b in C.BANDAS:
            prev = ros.reindex(sd_ - pd.Timedelta(days=(a + b) // 2)).values
            st = np.where(prev > C.ROS_ALTO, "alto", np.where(prev < C.ROS_BAJO, "bajo", "normal"))
            k = st == estado
            ajuste[str(thr)].append(round(float(bajo[k].mean() / bajo.mean()), 3) if k.sum() > 300 else 1.0)
    pct = float((ros < ros_hoy).mean())
    oni_ult = oni.dropna()
    return dict(bins=C.BINS, tabla=tabla, bandas=C.BANDAS, ajuste_rosario=ajuste,
                rosario=dict(valor=ros_hoy, fecha=str(ros.index[-1].date()), estado=estado, percentil=round(pct, 3)),
                oni=dict(valor=float(oni_ult.iloc[-1]), mes=str(oni_ult.index[-1].date())))


def refrescar_rios():
    """Agrega a data/rosario.json y data/concordia.json los últimos 45 días del INA."""
    hoy = pd.Timestamp.now()
    for nombre, sid in [("rosario", 34), ("concordia", 79)]:
        f = ROOT / f"data/{nombre}.json"
        url = (f"https://alerta.ina.gob.ar/a5/obs/puntual/series/{sid}/observaciones?timestart="
               f"{(hoy - pd.Timedelta(days=45)):%Y-%m-%d}&timeend={(hoy + pd.Timedelta(days=1)):%Y-%m-%d}&format=json")
        try:
            nuevo = json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as ex:
            print("no pude refrescar", nombre, ex, flush=True)
            continue
        d = {o["timestart"]: o for o in json.load(open(f))}
        d.update({o["timestart"]: o for o in nuevo})
        f.write_text(json.dumps(sorted(d.values(), key=lambda o: o["timestart"])))
        print(nombre, "hasta", max(d), flush=True)


def main():
    if socket.gethostname() != "raspberrypi":
        os.system(f"rsync -a mathigatti@raspberrypi.local:altura-del-rio/datos.db {ROOT}/datos.db")
    refrescar_rios()
    m = entrenar()
    progreso(10, 11)
    o, _ = M.cargar()
    ob = o.dropna()
    coef, *_ = np.linalg.lstsq(P.base_marea(ob.index), ob.values, rcond=None)
    orden = [(float(p), bool(p < 30 and k not in ("S2", "S1", "T2", "P1"))) for k, p in {**M.PERIODOS, **P.EXTRA}.items()]

    fc = bajar_pronostico()
    t, nivel, prob, sd, ahora = pronosticar(m, fc)
    vi = fc["rdp_medio"].reindex(t)
    sam = fc["samborombon"].reindex(t)
    r = riesgo_historico(coef)
    con = pd.DataFrame(json.load(open(ROOT / "data/concordia.json")))

    obs = o[o.index >= "2024-01-01"]
    ep = lambda ts: int(pd.Timestamp(ts).timestamp() // 3600)
    data = dict(
        generado=pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"), emision=ahora.isoformat(),
        umbral=U,
        marea=dict(periodos=orden, coef=[round(float(c), 6) for c in coef], nodal_h=P.NODAL,
                   check=[[ep(x), round(float((P.base_marea(pd.DatetimeIndex([x])) @ coef)[0]), 4)] for x in ["2026-11-28 12:00", "2027-03-01 03:00"]]),
        obs=dict(h0=ep(obs.index[0]), v=[None if np.isnan(x) else round(float(x), 2) for x in obs.values]),
        prono=dict(h0=ep(t[0]), nivel=[round(float(x), 3) for x in nivel], prob=[round(float(x), 3) for x in prob],
                   sd=[round(float(x), 3) for x in sd],
                   viento_kmh=[round(float(x) * 3.6, 1) for x in vi.vel], viento_dir=[None if np.isnan(x) else int(x) for x in vi.dir],
                   sambo_kmh=[round(float(x) * 3.6, 1) for x in sam.vel], sambo_dir=[None if np.isnan(x) else int(x) for x in sam.dir]),
        riesgo=r,
        concordia=dict(valor=float(con.valor.iloc[-1]), fecha=con.timestart.iloc[-1][:10]),
        feriados=FERIADOS,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False).replace("NaN", "null"))
    print("ok", OUT / "data.json", (OUT / "data.json").stat().st_size // 1024, "KB")
    progreso(11, 11)


if __name__ == "__main__":
    main()
