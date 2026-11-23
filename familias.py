"""Mismas variables que explicar.py (escalón 7), distintas familias de modelos. Train 2011-2018, valida 2019-2020, test 2021-2026."""
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import explicar as E
import modelo as M

ROOT = Path(__file__).parent
PROG = Path.home() / ".claude/progress/familias.json"
VAL = pd.Timestamp("2019-01-01")


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "familias", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def metricas(nombre, y, p):
    e = y - p
    b = y < 0.4
    return dict(modelo=nombre, R2=1 - e.var() / y.var(), MAE_cm=100 * e.abs().mean(),
                err_p90_cm=100 * e.abs().quantile(0.9), sesgo_bajantes_cm=100 * e[b].mean(),
                sesgo_crecidas_cm=100 * e[y > 2.2].mean(), detecta_bajantes=float(((p < 0.5) & b).sum() / b.sum()))


def main():
    F, g = E.armar()
    cols = [c for k, v in g.items() if k != "rio_24h_antes" for c in v]
    F = F.dropna(subset=["y", "marea"])
    tr, va, te = F[F.index < VAL], F[(F.index >= VAL) & (F.index < M.CORTE)], F[F.index >= M.CORTE]
    trva = F[F.index < M.CORTE]
    print(f"train {len(tr)}, val {len(va)}, test {len(te)}, features {len(cols)}", flush=True)
    n, i, preds, filas = 6, 0, {}, []

    # 1. Ridge: elige alpha en validación
    best = None
    for a in [1, 10, 100, 1000]:
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=a)).fit(tr[cols], tr.y)
        e = np.abs(va.y - m.predict(va[cols])).mean()
        best = min(best or (e, a), (e, a))
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=best[1])).fit(trva[cols], trva.y)
    preds["ridge (lineal)"] = m.predict(te[cols])
    i += 1; progreso(i, n)

    # 2. LightGBM normal (referencia) y 3. con modelos lineales en las hojas
    base = dict(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    preds["lightgbm"] = lgb.LGBMRegressor(**base).fit(trva[cols], trva.y).predict(te[cols])
    i += 1; progreso(i, n)
    imp = SimpleImputer(strategy="median").fit(trva[cols])
    Xtr, Xte = imp.transform(trva[cols]), imp.transform(te[cols])
    preds["lightgbm linear_tree"] = lgb.LGBMRegressor(**{**base, "num_leaves": 31}, linear_tree=True, linear_lambda=1.0).fit(Xtr, trva.y).predict(Xte)
    i += 1; progreso(i, n)

    # 4. Red neuronal
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=512,
                                                       learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                                                       validation_fraction=0.15, n_iter_no_change=10, random_state=0))
    preds["red neuronal (MLP)"] = mlp.fit(Xtr, trva.y).predict(Xte)
    i += 1; progreso(i, n)

    preds["promedio lineal_tree + MLP + ridge"] = np.mean([preds["lightgbm linear_tree"], preds["red neuronal (MLP)"], preds["ridge (lineal)"]], axis=0)
    preds["promedio todos"] = np.mean([v for k, v in preds.items() if not k.startswith("promedio")], axis=0)
    i += 1; progreso(i, n)

    for k, p in preds.items():
        filas.append(metricas(k, te.y, p))
    r = pd.DataFrame(filas)
    r.to_csv(ROOT / "results/familias.csv", index=False)
    pd.DataFrame(preds, index=te.index).assign(y=te.y).to_parquet(ROOT / "results/familias_pred.parquet")
    print(r.round(3).to_string(index=False))
    progreso(n, n)


if __name__ == "__main__":
    main()
