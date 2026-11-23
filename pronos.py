"""¿Cuánto del R² con viento perfecto (ERA5) sobrevive con pronósticos de viento reales archivados?

Open-Meteo Previous Runs (ECMWF IFS 0.25°, desde 2024-02-15): viento/presión pronosticados 1..7 días antes.
Entrena con ERA5 hasta 2024-02 ("perfect prog"); en test arma, para cada antelación d, las mismas features que
explicar.py pero con el viento posterior a la emisión tomado del pronóstico de la antelación correspondiente.
Variantes: sin río / + nivel de San Fernando al momento de emitir. Escribe results/pronos.csv.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import explicar as E
import modelo as M

ROOT = Path(__file__).parent
DIR = ROOT / "data/prev_runs"
PROG = Path.home() / ".claude/progress/pronos.json"
INI, FIN = "2024-02-15", "2026-11-21"
CORTE = pd.Timestamp(INI)
LEADS = [1, 2, 3, 4, 5, 7]
PUNTOS = E.PUNTOS + E.SUR


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "pronos", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def bajar():
    DIR.mkdir(exist_ok=True)
    for p in PUNTOS:
        f = DIR / f"{p}.csv"
        if f.exists():
            continue
        lat, lon = pd.read_csv(ROOT / f"data/era5/{p}.csv", nrows=1)[["latitude", "longitude"]].iloc[0]
        vs = [f"{v}{s}" for v in ["wind_speed_10m", "wind_direction_10m", "pressure_msl"]
              for s in [""] + [f"_previous_day{k}" for k in range(1, 8)]]
        url = (f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&hourly={','.join(vs)}&models=ecmwf_ifs025&start_date={INI}&end_date={FIN}&wind_speed_unit=ms")
        h = json.load(urllib.request.urlopen(url, timeout=300))["hourly"]
        pd.DataFrame(h).to_csv(f, index=False)
        print("bajado", p, flush=True)
        time.sleep(2)


def fuentes(idx):
    """S[p][k] = DataFrame u, v, vel, pres; k=0 ERA5 (pasado conocido), k>=1 pronóstico con k días de antelación."""
    S = {}
    for p in PUNTOS:
        era = E.era5(p, idx)
        d = pd.read_csv(DIR / f"{p}.csv", parse_dates=["time"]).set_index("time").reindex(idx)
        S[p] = {0: era}
        for k in range(1, 8):
            s = f"_previous_day{k}"
            vel, dr, pr = d[f"wind_speed_10m{s}"], d[f"wind_direction_10m{s}"], d[f"pressure_msl{s}"]
            rad = np.deg2rad(dr)
            w = pd.DataFrame({"u": -vel * np.sin(rad), "v": -vel * np.cos(rad), "vel": vel, "pres": pr}).interpolate(limit=6)
            S[p][k] = w.where(w.notna(), era)  # huecos del archivo: cae a ERA5 (pocos)
    return S


def armar_lead(F0, S, d, idx):
    """Reemplaza las columnas de viento/presión de F0 por la mezcla ERA5/pronóstico vista desde una emisión d días antes."""
    F = F0.copy()
    H = 24 * d

    def fuente(L):
        return min(7, int(np.ceil((H - L) / 24))) if H - L > 0 else 0

    def lag(p, var, L):
        return S[p][fuente(L)][var].shift(L)

    def media(p, x, n):  # media de x (función de la fuente) en las últimas n horas, por tramos de fuente
        tot = pd.Series(0.0, index=idx)
        L = 0
        while L < n:
            k = fuente(L)
            fin = min(n, H - 24 * (k - 1)) if k > 0 else n
            w = fin - L
            tot = tot + x(S[p][k]).rolling(w).sum().shift(L)
            L = fin
        return tot / n

    for p in PUNTOS:
        lags = E.LAGS + ([96, 120] if p in E.SUR else [])
        for L in lags:
            F[f"{p}_u_{L}"], F[f"{p}_v_{L}"] = lag(p, "u", L), lag(p, "v", L)
        for n in [6, 12, 24, 48, 72]:
            F[f"{p}_su_m{n}"] = media(p, lambda w: w.u * w.vel, n)
            F[f"{p}_sv_m{n}"] = media(p, lambda w: w.v * w.vel, n)
    for p in E.PUNTOS:
        F[f"{p}_p"] = lag(p, "pres", 0)
        F[f"{p}_dp12"] = lag(p, "pres", 0) - lag(p, "pres", 12)
    for a, b in [("plataforma_sur", "delta"), ("plataforma_este", "delta"), ("boca", "delta"),
                 ("samborombon", "rdp_medio"), ("plataforma_sur", "plataforma_este")]:
        for L in [0, 6, 12, 24]:
            F[f"grad_{a}_{b}_{L}"] = lag(a, "pres", L) - lag(b, "pres", L)
    for p in E.SUR:
        F[f"{p}_p"] = lag(p, "pres", 0)
        for L in [0, 12, 24, 48]:
            F[f"grad_{p}_delta_{L}"] = lag(p, "pres", L) - lag("delta", "pres", L)
    return F


def con_rio(F, o, d):
    F = F.copy()
    F["sf_emision"] = o.shift(24 * d)
    F["sf_m72"] = o.shift(24 * d).rolling(72, min_periods=24).mean()
    F["res_emision"] = F.sf_emision - F.marea.shift(24 * d)
    return F


def metricas(y, p, pb, clim, **kw):
    e = y - p
    b = (y < 0.4).values
    return dict(**kw, R2=1 - e.var() / y.var(), MAE_cm=100 * e.abs().mean(),
                err_p90_cm=100 * e.abs().quantile(0.9), sesgo_bajantes_cm=100 * e[b].mean(),
                detecta_bajantes=float(((p < 0.5) & b).sum() / b.sum()),
                falsas_alarmas=float(((p < 0.5) & ~b).sum() / max(1, (p < 0.5).sum())),
                BSS_bajo40=1 - np.mean((pb - b) ** 2) / np.mean((clim - b) ** 2))


def main():
    bajar()
    F0, g = E.armar()
    o, _ = M.cargar()
    idx = F0.index
    # Rosario/Uruguay: atrasados una semana para que valgan para cualquier antelación <= 7 d
    for c in ["ros", "ros_d7", "ros_d30", "con", "con_d7"]:
        F0[c] = F0[c].shift(24 * 7)
    cols = [c for k, v in g.items() if k != "rio_24h_antes" for c in v]
    cols_rio = cols + ["sf_emision", "sf_m72", "res_emision"]
    F0 = F0.dropna(subset=["y", "marea"])
    idx = F0.index
    S = fuentes(idx)
    tr_mask, te_mask = idx < CORTE, idx >= CORTE
    tr = F0[tr_mask]
    te_y = F0.y[te_mask]
    clim = te_y.index.month.map(tr.groupby(tr.index.month).y.apply(lambda s: (s < 0.4).mean())).values
    params = dict(n_estimators=1200, learning_rate=0.04, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    n, i, filas = 3 + 2 * len(LEADS), 0, []

    def fit(X, y):
        return lgb.LGBMRegressor(**params).fit(X, y), lgb.LGBMClassifier(**params).fit(X, (y < 0.4).astype(int))

    # tabla de mareas sola (referencia)
    r, c = fit(tr[g["marea"]], tr.y)
    Xt = F0.loc[te_mask, g["marea"]]
    filas.append(metricas(te_y, r.predict(Xt), c.predict_proba(Xt)[:, 1], clim, variante="tabla de mareas", lead_d=-1))
    i += 1; progreso(i, n)

    # sin río: un solo modelo; cambia solo el viento de entrada
    r, c = fit(tr[cols], tr.y)
    i += 1; progreso(i, n)
    Xt = F0.loc[te_mask, cols]
    filas.append(metricas(te_y, r.predict(Xt), c.predict_proba(Xt)[:, 1], clim, variante="sin río", lead_d=0))
    Fd = {}
    for d in LEADS:
        Fd[d] = armar_lead(F0, S, d, idx)
        Xt = Fd[d].loc[te_mask, cols]
        filas.append(metricas(te_y, r.predict(Xt), c.predict_proba(Xt)[:, 1], clim, variante="sin río", lead_d=d))
        i += 1; progreso(i, n)
        print(pd.DataFrame(filas).round(3).tail(1).to_string(index=False), flush=True)

    # con el río al momento de emitir: un modelo por antelación (el retardo del río depende de d)
    for d in LEADS:
        Ftr = con_rio(F0, o, d)
        r, c = fit(Ftr.loc[tr_mask, cols_rio], tr.y)
        for fuente, F in [("ERA5", Ftr), ("ECMWF", con_rio(Fd[d], o, d))]:
            Xt = F.loc[te_mask, cols_rio]
            filas.append(metricas(te_y, r.predict(Xt), c.predict_proba(Xt)[:, 1], clim,
                                  variante=f"con río, viento {fuente}", lead_d=d))
        i += 1; progreso(i, n)
        print(pd.DataFrame(filas).round(3).tail(2).to_string(index=False), flush=True)

    res = pd.DataFrame(filas)
    res.to_csv(ROOT / "results/pronos.csv", index=False)
    print(res.round(3).to_string(index=False))
    progreso(n, n)


if __name__ == "__main__":
    main()
