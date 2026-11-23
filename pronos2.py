"""Pronóstico real (viento ECMWF archivado) con la marea mejorada de empujar.py + corrección de sesgo.

Train ERA5 < 2024-02-15. Calibración (isotónica por antelación) con pronósticos reales 2024-02-15 → 2025-02-15.
Evaluación 2025-02-15 → 2026-11 (no vista por el calibrador). Escribe results/pronos2.csv y results/pronos2_pred.parquet.
"""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

import empujar as P
import explicar as E
import modelo as M
import pronos as PR

ROOT = Path(__file__).parent
PROG = Path.home() / ".claude/progress/pronos2.json"
CAL = pd.Timestamp("2025-02-15")
LEADS = [0, 1, 2, 3, 4, 5, 7]
U = 0.4


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "pronos2", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def marea_feats(F, o):
    idx = F.index
    tr0 = o[o.index < M.CORTE].dropna()
    coef, *_ = np.linalg.lstsq(P.base_marea(tr0.index), tr0.values, rcond=None)
    m2 = pd.Series(P.base_marea(idx) @ coef, index=idx)
    cols = ["marea2"]
    F["marea2"] = m2
    for k in [-6, -4, -2, -1, 1, 2, 4, 6]:
        F[f"marea2_{k}"] = m2.shift(k)
        cols.append(f"marea2_{k}")
    h = (idx - pd.Timestamp("2000-01-01")).total_seconds().values / 3600
    for c in P.FASES:
        F[f"fase_{c}_c"], F[f"fase_{c}_s"] = np.cos(2 * np.pi * h / M.PERIODOS[c]), np.sin(2 * np.pi * h / M.PERIODOS[c])
        cols += [f"fase_{c}_c", f"fase_{c}_s"]
    F["marea2_rango25"] = m2.rolling(25, center=True).max() - m2.rolling(25, center=True).min()
    return cols + ["marea2_rango25"]


def metricas(y, p, pb, clim, **kw):
    e = y - p
    b = (y < U).values
    alarma = pb > 0.5
    return dict(**kw, R2=1 - e.var() / y.var(), MAE_cm=100 * e.abs().mean(),
                sesgo_bajantes_cm=100 * e[b].mean(), sesgo_crecidas_cm=100 * e[(y > 2.2).values].mean(),
                detecta_bajantes=float((alarma & b).sum() / b.sum()),
                falsas_alarmas=float((alarma & ~b).sum() / max(1, alarma.sum())),
                BSS_bajo40=1 - np.mean((pb - b) ** 2) / np.mean((clim - b) ** 2),
                prob_media=float(np.mean(pb)), frec_real=float(b.mean()))


def main():
    F0, g = E.armar()
    o, _ = M.cargar()
    for c in ["ros", "ros_d7", "ros_d30", "con", "con_d7"]:
        F0[c] = F0[c].shift(24 * 7)
    tide = marea_feats(F0, o)
    cols = [c for k, v in g.items() if k != "rio_24h_antes" for c in v] + tide
    F0 = F0.dropna(subset=["y", "marea"])
    idx = F0.index
    S = PR.fuentes(idx)
    trm = idx < PR.CORTE
    tr = F0[trm]
    calm, evm = (idx >= PR.CORTE) & (idx < CAL), idx >= CAL
    y_cal, y_ev = F0.y[calm], F0.y[evm]
    clim_m = tr.groupby(tr.index.month).y.apply(lambda s: (s < U).mean())
    clim_ev = y_ev.index.month.map(clim_m).values

    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, linear_tree=True, linear_lambda=1.0, verbose=-1)
    cparams = {**params, "linear_tree": False, "num_leaves": 63}
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    Xtr = imp.transform(tr[cols])
    peso = np.where(tr.y < 0.6, 4.0, np.where(tr.y > 2.0, 2.0, 1.0))
    n, i = 3 + len(LEADS), 0
    reg = lgb.LGBMRegressor(**params).fit(Xtr, tr.y); i += 1; progreso(i, n)
    regw = lgb.LGBMRegressor(**params).fit(Xtr, tr.y, sample_weight=peso); i += 1; progreso(i, n)
    cla = lgb.LGBMClassifier(**cparams).fit(Xtr, (tr.y < U).astype(int)); i += 1; progreso(i, n)

    filas, out = [], {}
    for d in LEADS:
        F = F0 if d == 0 else PR.armar_lead(F0, S, d, idx)
        Xc, Xe = imp.transform(F.loc[calm, cols]), imp.transform(F.loc[evm, cols])
        pc, pe = reg.predict(Xc), reg.predict(Xe)
        pwc, pwe = regw.predict(Xc), regw.predict(Xe)
        qc, qe = cla.predict_proba(Xc)[:, 1], cla.predict_proba(Xe)[:, 1]
        # calibradores isotónicos ajustados con pronósticos reales del año de calibración
        niv = IsotonicRegression(out_of_bounds="clip").fit(pc, y_cal)
        prob_reg = IsotonicRegression(out_of_bounds="clip", increasing=False, y_min=0, y_max=1).fit(pc, (y_cal < U).astype(float))
        prob_cla = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(qc, (y_cal < U).astype(float))
        pcal = niv.predict(pe)
        cand = {
            "crudo": (pe, qe),
            "pesos bajantes": (pwe, qe),
            "calibrado (isotónica)": (pcal, prob_reg.predict(pe)),
            "calibrado, prob del clasificador": (pcal, prob_cla.predict(qe)),
        }
        for k, (p, pb) in cand.items():
            filas.append(metricas(y_ev, p, pb, clim_ev, variante=k, lead_d=d))
        out[f"p_{d}"], out[f"prob_{d}"] = pcal, prob_reg.predict(pe)
        i += 1; progreso(i, n)
        print(pd.DataFrame(filas).round(3).tail(4).to_string(index=False), flush=True)

    r = pd.DataFrame(filas)
    r.to_csv(ROOT / "results/pronos2.csv", index=False)
    pd.DataFrame(out, index=y_ev.index).assign(y=y_ev).to_parquet(ROOT / "results/pronos2_pred.parquet")
    print(r.round(3).to_string(index=False))
    progreso(n, n)


if __name__ == "__main__":
    main()
