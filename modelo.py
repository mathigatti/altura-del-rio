"""Modelo probabilístico de altura en San Fernando a 6 h – 14 días.

Entrena 2011-2020, evalúa 2021-2026. Escribe results/metricas.csv.
"""
import json
import os
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CORTE = pd.Timestamp("2021-01-01")
LEADS = [6, 12, 24, 48, 72, 96, 120, 168, 240, 336]
UMBRALES = [0.3, 0.4, 0.5]
QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
PERIODOS = {  # horas
    "M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K2": 11.96723606, "2N2": 12.90537,
    "MU2": 12.8717576, "K1": 23.93447213, "O1": 25.81933871, "P1": 24.06588766, "Q1": 26.868350,
    "M4": 6.210300601, "MS4": 6.103339, "MN4": 6.269173724, "M6": 4.140200401, "M3": 8.280400802,
    "MM": 661.3111655, "MSF": 354.3670666, "SA": 8766.15265, "SSA": 4383.076325,
}


def cargar():
    c = sqlite3.connect(ROOT / "datos.db")
    o = pd.read_sql("select t, valor from obs_sf", c, parse_dates=["t"])
    o["t"] = o.t.dt.floor("h")
    o = o.groupby("t").valor.mean()
    idx = pd.date_range("2011-01-01", o.index.max(), freq="h")
    o = o.reindex(idx)
    o[(o < -1.5) | (o > 5)] = np.nan

    w = pd.read_csv(ROOT / "data/viento_era5.csv", skiprows=3, parse_dates=["time"]).set_index("time")
    w.columns = ["vel", "dir", "pres"]
    rad = np.deg2rad(w.dir)
    w["u"] = -w.vel * np.sin(rad)  # componente hacia el este
    w["v"] = -w.vel * np.cos(rad)  # componente hacia el norte
    w = w[["u", "v", "pres"]].reindex(idx).interpolate(limit=6)
    r = pd.DataFrame(json.load(open(ROOT / "data/rosario.json")))
    r["t"] = pd.to_datetime(r.timestart, utc=True).dt.tz_localize(None)
    ros = r.groupby(r.t.dt.floor("D")).valor.mean()
    # dato diario disponible recién al día siguiente: se corre 1 día para no filtrar el futuro
    w["ros"] = ros.shift(1, freq="D").reindex(idx, method="ffill", limit=24 * 5)
    return o, w


def base_armonica(t):
    h = (t - pd.Timestamp("2000-01-01")).total_seconds().values / 3600
    cols = [np.ones_like(h)]
    for p in PERIODOS.values():
        cols += [np.cos(2 * np.pi * h / p), np.sin(2 * np.pi * h / p)]
    return np.column_stack(cols)


def ajustar_marea(o):
    tr = o[o.index < CORTE].dropna()
    coef, *_ = np.linalg.lstsq(base_armonica(tr.index), tr.values, rcond=None)
    marea = pd.Series(base_armonica(o.index) @ coef, index=o.index)
    return marea


def features(o, w, marea):
    r = o - marea
    rl = r.ffill(limit=6)  # último residuo disponible
    f = pd.DataFrame(index=o.index)
    f["r0"] = rl
    f["r_d6"] = rl - rl.shift(6)
    for n in [24, 72, 168]:
        f[f"r_m{n}"] = r.rolling(n, min_periods=n // 3).mean()
    f["nivel_m720"] = o.rolling(720, min_periods=240).mean()
    f["nivel_min72"] = o.rolling(72, min_periods=24).min()
    f["u0"], f["v0"], f["p0"] = w.u, w.v, w.pres
    for n in [24, 72]:
        f[f"u_m{n}"] = w.u.rolling(n, min_periods=n // 2).mean()
        f[f"v_m{n}"] = w.v.rolling(n, min_periods=n // 2).mean()
    f["dp24"] = w.pres - w.pres.shift(24)
    f["ros0"] = w.ros
    f["ros_d7"] = w.ros - w.ros.shift(24 * 7)
    f["ros_d30"] = w.ros - w.ros.shift(24 * 30)
    return f


def dataset(o, w, marea, f):
    emisiones = o.index[::3]
    wu24 = w.u.rolling(24, min_periods=12).mean()
    wv24 = w.v.rolling(24, min_periods=12).mean()
    filas = []
    for L in LEADS:
        t0 = emisiones[emisiones + pd.Timedelta(hours=L) <= o.index[-1]]
        T = t0 + pd.Timedelta(hours=L)
        d = f.loc[t0].reset_index(drop=True)
        d["lead"] = L
        d["t0"], d["T"] = t0, T
        d["marea_T"] = marea.loc[T].values
        d["marea_T_min6"] = marea.rolling(13, center=True).min().loc[T].values
        doy = T.dayofyear.values
        d["doy_s"], d["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        d["hora_T"] = T.hour.values
        d["uT"], d["vT"] = w.u.loc[T].values, w.v.loc[T].values  # solo para el techo (viento perfecto)
        d["uT24"], d["vT24"] = wu24.loc[T].values, wv24.loc[T].values
        d["y"] = o.loc[T].values
        filas.append(d)
    return pd.concat(filas, ignore_index=True).dropna(subset=["y", "r0"])


VIENTO_FUTURO = ["uT", "vT", "uT24", "vT24"]
ROSARIO = ["ros0", "ros_d7", "ros_d30"]
PROG = Path.home() / ".claude/progress/modelo-rosario.json"
_hecho = [0]


def progreso(total):
    _hecho[0] += 1
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "modelo-rosario", "done": _hecho[0], "total": total, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{_hecho[0]}/{total}] modelos entrenados", flush=True)


def entrenar(tr, cols, objetivo, **kw):
    p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, **kw)
    m = lgb.LGBMRegressor(**p) if objetivo != "binary" else lgb.LGBMClassifier(**p)
    m.fit(tr[cols], tr["_y"] if objetivo == "binary" else tr["y"])
    return m


def brier(p, y):
    return np.mean((p - y) ** 2)


def main():
    o, w = cargar()
    marea = ajustar_marea(o)
    r = (o - marea).dropna()
    print(f"marea armónica: R² test = {1 - r[r.index >= CORTE].var() / o[o.index >= CORTE].var():.2f}, "
          f"desvío residuo = {r.std():.2f} m")
    f = features(o, w, marea)
    ds = dataset(o, w, marea, f)
    tr, te = ds[ds["T"] < CORTE - pd.Timedelta(days=14)].copy(), ds[ds.t0 >= CORTE].copy()
    print(f"train {len(tr)} filas, test {len(te)} filas")

    base = [c for c in ds.columns if c not in ["t0", "T", "y", *VIENTO_FUTURO, *ROSARIO]]
    variantes = {"modelo": base, "modelo_rosario": base + ROSARIO,
                 "techo_viento_perfecto": base + ROSARIO + VIENTO_FUTURO}
    total = len(UMBRALES) * len(variantes) + 2 * len(QS)

    res_train = (o - marea)[o.index < CORTE].dropna().values
    mes_tr = tr.groupby(tr["T"].dt.month)

    filas = []
    for thr in UMBRALES:
        tr["_y"] = (tr.y < thr).astype(int)
        yte = (te.y < thr).astype(int).values
        clim = te["T"].dt.month.map(mes_tr.apply(lambda g: (g.y < thr).mean())).values
        marea_clim = np.array([(res_train < thr - m).mean() for m in te.marea_T.values])
        preds = {"climatologia": clim, "tabla_mareas": marea_clim}
        for nombre, cols in variantes.items():
            preds[nombre] = entrenar(tr, cols, "binary").predict_proba(te[cols])[:, 1]
            progreso(total)
        for L in LEADS:
            mk = te.lead.values == L
            bc = brier(clim[mk], yte[mk])
            for nombre, p in preds.items():
                filas.append(dict(umbral=thr, lead_h=L, variante=nombre, freq=yte[mk].mean(),
                                  brier=brier(p[mk], yte[mk]), bss=1 - brier(p[mk], yte[mk]) / bc))
        te[f"p_{thr}"] = preds["modelo"]
        te[f"p_{thr}_ros"] = preds["modelo_rosario"]
        te[f"p_{thr}_techo"] = preds["techo_viento_perfecto"]

    for suf, cols in [("", base), ("_ros", base + ROSARIO)]:
        for q in QS:
            te[f"q{q}{suf}"] = entrenar(tr, cols, "quantile", objective="quantile", alpha=q).predict(te[cols])
            progreso(total)

    met = pd.DataFrame(filas)
    met.to_csv(OUT / "metricas.csv", index=False)
    te.drop(columns=["_y"], errors="ignore").to_parquet(OUT / "pred_test.parquet")

    print("\nBrier skill score vs climatología (1 = perfecto, 0 = igual que lo normal del mes)")
    for thr in UMBRALES:
        t = met[met.umbral == thr].pivot(index="lead_h", columns="variante", values="bss")
        print(f"\nP(nivel < {thr} m)  frecuencia en test: {met[met.umbral == thr].freq.mean():.1%}")
        print(t[["tabla_mareas", "modelo", "modelo_rosario", "techo_viento_perfecto"]].round(2).to_string())

    print("\nBandas de cuantiles (sin viento futuro), sin Rosario | con Rosario:")
    g = te.groupby("lead")
    tab = {}
    for suf, nm in [("", "sin"), ("_ros", "con")]:
        tab[f"MAE_{nm}"] = g.apply(lambda d: (d.y - d[f"q0.5{suf}"]).abs().mean())
        tab[f"cob10_90_{nm}"] = g.apply(lambda d: ((d.y >= d[f"q0.1{suf}"]) & (d.y <= d[f"q0.9{suf}"])).mean())
        tab[f"ancho_{nm}"] = g.apply(lambda d: (d[f"q0.9{suf}"] - d[f"q0.1{suf}"]).mean())
    print(pd.DataFrame(tab).round(2).to_string())


if __name__ == "__main__":
    main()
