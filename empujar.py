"""Empujar el R² de reconstrucción (viento perfecto ERA5) por encima de 0,875.

Escalones sobre las features de explicar.py (escalón 7), LightGBM linear_tree, train <2021, test >=2021:
  A base
  B + marea mejorada (más constituyentes + modulación nodal 18,6 años) y fases de cada constituyente
  C + grilla ERA5 de 30 puntos resumida en EOFs de la tensión del viento (con retardos y medias)
  D igual que C pero aprendiendo el residuo y - marea
Escribe results/empujar.csv y results/empujar_pred.parquet.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import explicar as E
import modelo as M

ROOT = Path(__file__).parent
GRID = ROOT / "data/era5_grilla"
PROG = Path.home() / ".claude/progress/empujar.json"
LATS = [-33.5, -35.0, -36.5, -38.0, -39.5, -41.0]
LONS = [-59.5, -58.0, -56.5, -55.0, -53.5]
EXTRA = {"2SM2": 11.6069, "L2": 12.1916, "T2": 12.0164, "NU2": 12.6260, "LDA2": 12.2218, "J1": 23.0985,
         "OO1": 22.3061, "S1": 24.0, "M1": 24.8412, "MK3": 8.1771, "SO3": 7.8, "2MK3": 8.3863,
         "2MS6": 4.0924, "2MN6": 4.1663, "MSN6": 4.0500, "M8": 3.1052, "MF": 327.8590}
NODAL = 18.6129 * 365.25 * 24
FASES = ["M2", "S2", "N2", "K1", "O1", "M4"]


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "empujar", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def bajar_grilla():
    GRID.mkdir(exist_ok=True)
    for la in LATS:
        for lo in LONS:
            f = GRID / f"{la}_{lo}.csv"
            if f.exists():
                continue
            url = ("https://archive-api.open-meteo.com/v1/archive?"
                   f"latitude={la}&longitude={lo}&start_date=2010-12-01&end_date=2026-11-20"
                   "&hourly=wind_speed_10m,wind_direction_10m,pressure_msl&models=era5&wind_speed_unit=ms")
            for intento in range(5):
                try:
                    h = json.load(urllib.request.urlopen(url, timeout=600))["hourly"]
                    break
                except Exception as ex:
                    print("reintento", la, lo, ex, flush=True)
                    time.sleep(60)
            else:
                raise RuntimeError(f"no pude bajar {la},{lo}")
            pd.DataFrame(h).to_csv(f, index=False)
            print("bajado", la, lo, flush=True)
            time.sleep(3)


def base_marea(t):
    h = (t - pd.Timestamp("2000-01-01")).total_seconds().values / 3600
    nod = 2 * np.pi * h / NODAL
    cols = [np.ones_like(h)]
    for nombre, p in {**M.PERIODOS, **EXTRA}.items():
        c, s = np.cos(2 * np.pi * h / p), np.sin(2 * np.pi * h / p)
        cols += [c, s]
        if p < 30 and nombre not in ("S2", "S1", "T2", "P1"):  # constituyentes lunares: f·cos(ωt+u) ≈ 1er orden en N
            cols += [c * np.cos(nod), c * np.sin(nod), s * np.cos(nod), s * np.sin(nod)]
    return np.column_stack(cols)


def eofs(idx, k=8):
    U, V = [], []
    for la in LATS:
        for lo in LONS:
            d = pd.read_csv(GRID / f"{la}_{lo}.csv", parse_dates=["time"]).set_index("time").reindex(idx).interpolate(limit=6)
            rad = np.deg2rad(d.wind_direction_10m)
            u, v = -d.wind_speed_10m * np.sin(rad), -d.wind_speed_10m * np.cos(rad)
            U.append(u * d.wind_speed_10m)
            V.append(v * d.wind_speed_10m)
    X = pd.concat(U + V, axis=1).ffill().bfill()
    tr = X.index < M.CORTE
    mu, sd = X[tr].mean(), X[tr].std()
    Z = ((X - mu) / sd).values
    _, sv, vt = np.linalg.svd(Z[tr], full_matrices=False)
    print("varianza explicada EOFs:", np.round((sv[:k] ** 2 / (sv ** 2).sum()), 3), flush=True)
    return pd.DataFrame(Z @ vt[:k].T, index=idx, columns=[f"eof{j}" for j in range(k)])


def metricas(nombre, y, p):
    e = y - p
    b = y < 0.4
    return dict(escalon=nombre, R2=1 - e.var() / y.var(), MAE_cm=100 * e.abs().mean(), RMSE_cm=100 * np.sqrt((e ** 2).mean()),
                sesgo_bajantes_cm=100 * e[b].mean(), sesgo_crecidas_cm=100 * e[y > 2.2].mean(),
                detecta_bajantes=float(((p < 0.5) & b).sum() / b.sum()))


def main():
    bajar_grilla()
    F, g = E.armar()
    o, _ = M.cargar()
    idx = F.index
    tr0 = o[o.index < M.CORTE].dropna()
    coef, *_ = np.linalg.lstsq(base_marea(tr0.index), tr0.values, rcond=None)
    marea2 = pd.Series(base_marea(idx) @ coef, index=idx)
    r2v = lambda m: 1 - (o - m).var() / o.var()
    print(f"marea vieja R2 test {r2v(F.marea[idx >= M.CORTE]):.3f}  nueva {r2v(marea2[idx >= M.CORTE]):.3f}", flush=True)

    tide = []
    F["marea2"] = marea2
    for k in [-6, -4, -2, -1, 1, 2, 4, 6]:
        F[f"marea2_{k}"] = marea2.shift(k)
    tide += ["marea2"] + [f"marea2_{k}" for k in [-6, -4, -2, -1, 1, 2, 4, 6]]
    h = (idx - pd.Timestamp("2000-01-01")).total_seconds().values / 3600
    for c in FASES:
        F[f"fase_{c}_c"], F[f"fase_{c}_s"] = np.cos(2 * np.pi * h / M.PERIODOS[c]), np.sin(2 * np.pi * h / M.PERIODOS[c])
        tide += [f"fase_{c}_c", f"fase_{c}_s"]
    F["marea2_rango25"] = marea2.rolling(25, center=True).max() - marea2.rolling(25, center=True).min()
    tide.append("marea2_rango25")

    Z = eofs(idx)
    grid = []
    for c in Z:
        for L in [0, 3, 6, 12, 24, 48]:
            F[f"{c}_{L}"] = Z[c].shift(L)
            grid.append(f"{c}_{L}")
        for n in [12, 24, 72, 168]:
            F[f"{c}_m{n}"] = Z[c].rolling(n).mean()
            grid.append(f"{c}_m{n}")

    base = [c for k, v in g.items() if k != "rio_24h_antes" for c in v]
    F = F.dropna(subset=["y", "marea"])
    trva, te = F[F.index < M.CORTE], F[F.index >= M.CORTE]
    escalones = [("A base (escalón 7)", base, False),
                 ("B + marea mejorada y fases", base + tide, False),
                 ("C + grilla EOF", base + tide + grid, False),
                 ("D C aprendiendo residuo", base + tide + grid, True)]
    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, linear_tree=True, linear_lambda=1.0, verbose=-1)
    filas, preds = [], {}
    for i, (nombre, cols, resid) in enumerate(escalones, 1):
        imp = SimpleImputer(strategy="median").fit(trva[cols])
        off_tr = trva.marea2 if resid else 0
        off_te = te.marea2 if resid else 0
        m = lgb.LGBMRegressor(**params).fit(imp.transform(trva[cols]), trva.y - off_tr)
        p = m.predict(imp.transform(te[cols])) + off_te
        preds[nombre] = np.asarray(p)
        filas.append(metricas(nombre, te.y, p))
        print(pd.DataFrame(filas).round(3).tail(1).to_string(index=False), flush=True)
        progreso(i, len(escalones))
    r = pd.DataFrame(filas)
    r.to_csv(ROOT / "results/empujar.csv", index=False)
    pd.DataFrame(preds, index=te.index).assign(y=te.y).to_parquet(ROOT / "results/empujar_pred.parquet")
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
