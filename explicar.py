"""¿Cuánto del nivel en San Fernando se explica con sus causas medidas (sin mirar el propio río)?

Ablación: marea → + viento 1 punto → + viento 7 puntos → + presión → + Rosario.
Train 2011-2020, test 2021-2026. Escribe results/explicar.csv.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import modelo as M

ROOT = Path(__file__).parent
PUNTOS = ["delta", "rdp_medio", "rdp_exterior", "boca", "samborombon", "plataforma_sur", "plataforma_este"]
SUR = ["mar_del_plata_off", "patagonia_norte", "patagonia_media"]
LAGS = [0, 3, 6, 9, 12, 18, 24, 36, 48, 72]
PROG = Path.home() / ".claude/progress/explicar.json"


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "explicar", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def era5(p, idx):
    w = pd.read_csv(ROOT / f"data/era5/{p}.csv", skiprows=3, parse_dates=["time"]).set_index("time")
    w.columns = ["vel", "dir", "pres"]
    rad = np.deg2rad(w.dir)
    w["u"], w["v"] = -w.vel * np.sin(rad), -w.vel * np.cos(rad)
    return w[["u", "v", "vel", "pres"]].reindex(idx).interpolate(limit=6)


def armar():
    o, w0 = M.cargar()
    idx = o.index
    tr = o[o.index < M.CORTE].dropna()
    coef, *_ = np.linalg.lstsq(M.base_armonica(tr.index), tr.values, rcond=None)
    marea = pd.Series(M.base_armonica(idx) @ coef, index=idx)

    g = {}
    F = pd.DataFrame(index=idx)
    F["marea"] = marea
    for k in [-3, -1, 1, 3]:
        F[f"marea_{k}"] = marea.shift(k)
    doy = idx.dayofyear
    F["doy_s"], F["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    g["marea"] = list(F.columns)

    W = {p: era5(p, idx) for p in PUNTOS + SUR}
    for p, w in W.items():
        lags = LAGS + ([96, 120] if p in SUR else [])
        cols = []
        su, sv = w.u * w.vel, w.v * w.vel  # tensión del viento ~ |w| w
        for L in lags:
            F[f"{p}_u_{L}"], F[f"{p}_v_{L}"] = w.u.shift(L), w.v.shift(L)
            cols += [f"{p}_u_{L}", f"{p}_v_{L}"]
        for n in [6, 12, 24, 48, 72]:
            F[f"{p}_su_m{n}"], F[f"{p}_sv_m{n}"] = su.rolling(n).mean(), sv.rolling(n).mean()
            cols += [f"{p}_su_m{n}", f"{p}_sv_m{n}"]
        g[f"viento_{p}"] = cols

    pc = []
    for p in PUNTOS:
        w = W[p]
        F[f"{p}_p"] = w.pres
        F[f"{p}_dp12"] = w.pres - w.pres.shift(12)
        pc += [f"{p}_p", f"{p}_dp12"]
    for a, b in [("plataforma_sur", "delta"), ("plataforma_este", "delta"), ("boca", "delta"),
                 ("samborombon", "rdp_medio"), ("plataforma_sur", "plataforma_este")]:
        d = W[a].pres - W[b].pres
        for L in [0, 6, 12, 24]:
            F[f"grad_{a}_{b}_{L}"] = d.shift(L)
            pc.append(f"grad_{a}_{b}_{L}")
    g["presion"] = pc
    ps = []
    for p in SUR:
        F[f"{p}_p"] = W[p].pres
        ps.append(f"{p}_p")
        for L in [0, 12, 24, 48]:
            F[f"grad_{p}_delta_{L}"] = (W[p].pres - W["delta"].pres).shift(L)
            ps.append(f"grad_{p}_delta_{L}")
    g["presion_sur"] = ps

    r = pd.DataFrame(json.load(open(ROOT / "data/concordia.json")))
    r["t"] = pd.to_datetime(r.timestart, utc=True).dt.tz_localize(None)
    con = r.groupby(r.t.dt.floor("D")).valor.mean().shift(1, freq="D").reindex(idx, method="ffill", limit=24 * 5)
    F["con"], F["con_d7"] = con, con - con.shift(24 * 7)
    g["uruguay"] = ["con", "con_d7"]

    F["ros"] = w0.ros
    F["ros_d7"] = w0.ros - w0.ros.shift(24 * 7)
    F["ros_d30"] = w0.ros - w0.ros.shift(24 * 30)
    g["rosario"] = ["ros", "ros_d7", "ros_d30"]

    F["sf_24"] = o.shift(24)  # solo como referencia: el río 24 h antes
    F["sf_m72"] = o.shift(24).rolling(72, min_periods=24).mean()
    g["rio_24h_antes"] = ["sf_24", "sf_m72"]
    F["y"] = o
    return F, g


def main():
    F, g = armar()
    F = F.dropna(subset=["y", "marea"])
    tr, te = F[F.index < M.CORTE], F[F.index >= M.CORTE]
    print(f"train {len(tr)} h, test {len(te)} h, features {F.shape[1] - 1}", flush=True)

    vientos_todos = [c for p in PUNTOS for c in g[f"viento_{p}"]]
    vientos_sur = [c for p in SUR for c in g[f"viento_{p}"]]
    escalones = [
        ("1 marea astronómica", g["marea"]),
        ("2 + viento 1 punto (medio RdlP)", g["marea"] + g["viento_rdp_medio"]),
        ("3 + viento 7 puntos", g["marea"] + vientos_todos),
        ("4 + presión", g["marea"] + vientos_todos + g["presion"]),
        ("5 + Rosario", g["marea"] + vientos_todos + g["presion"] + g["rosario"]),
        ("6 + viento y presión del sur (MdP, Patagonia)", g["marea"] + vientos_todos + g["presion"] + g["rosario"] + vientos_sur + g["presion_sur"]),
        ("7 + río Uruguay (Concordia)", g["marea"] + vientos_todos + g["presion"] + g["rosario"] + vientos_sur + g["presion_sur"] + g["uruguay"]),
        ("8 + río 24 h antes (referencia)", g["marea"] + vientos_todos + g["presion"] + g["rosario"] + vientos_sur + g["presion_sur"] + g["uruguay"] + g["rio_24h_antes"]),
    ]
    n = len(escalones) + 1
    filas, var_y = [], te.y.var()
    clim = te.index.month.map(tr.groupby(tr.index.month).y.apply(lambda s: (s < 0.4).mean())).values
    yb = (te.y < 0.4).values
    bclim = np.mean((clim - yb) ** 2)
    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    for i, (nombre, cols) in enumerate(escalones, 1):
        m = lgb.LGBMRegressor(**params).fit(tr[cols], tr.y)
        p = m.predict(te[cols])
        res = te.y - p
        c = lgb.LGBMClassifier(**params).fit(tr[cols], (tr.y < 0.4).astype(int))
        pb = c.predict_proba(te[cols])[:, 1]
        filas.append(dict(escalon=nombre, n_features=len(cols), R2=1 - res.var() / var_y,
                          MAE_cm=100 * res.abs().mean(), RMSE_cm=100 * np.sqrt((res ** 2).mean()),
                          err_p90_cm=100 * res.abs().quantile(0.9),
                          BSS_bajo40=1 - np.mean((pb - yb) ** 2) / bclim,
                          detecta_bajantes=float(((p < 0.5) & yb).sum() / yb.sum())))
        if i == 7:
            te.assign(pred=p, p_bajo=pb)[["y", "pred", "p_bajo"]].to_parquet(ROOT / "results/explicar_pred.parquet")
            imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
            grupos = {k: imp.reindex(v).sum() for k, v in g.items() if k != "rio_24h_antes"}
            print("importancia por grupo (gain, %):", {k: round(100 * v / imp.sum(), 1) for k, v in grupos.items()}, flush=True)
        progreso(i, n)
    r = pd.DataFrame(filas)
    r.to_csv(ROOT / "results/explicar.csv", index=False)
    print(r.round(3).to_string(index=False))

    # errores por régimen con el modelo 5
    d = pd.read_parquet(ROOT / "results/explicar_pred.parquet")
    d["err"] = d.y - d.pred
    d["rango"] = pd.cut(d.y, [-2, 0.4, 0.7, 1.5, 2.2, 5])
    print("\nerror por rango de nivel real (modelo 7):")
    print(d.groupby("rango", observed=True).err.agg(n="count", sesgo_cm=lambda s: 100 * s.mean(),
                                                    MAE_cm=lambda s: 100 * s.abs().mean()).round(1).to_string())
    progreso(n, n)


if __name__ == "__main__":
    main()
