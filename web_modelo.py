"""Exporta el modelo para la web (todo corre en el navegador): site/model.json + site/test_vector.json.

Entrena una versión liviana del modelo de pronos2 (300 árboles, lr 0.1) para que el JSON pese poco,
la calibra por antelación (isotónica, pronósticos ECMWF archivados 2024-02 → 2026-11) y exporta:
árboles, medianas del imputador, columnas, coeficientes de marea, calibradores, tabla de riesgo histórico.
El vector de prueba guarda entradas crudas + features + predicciones de Python para verificar el JS.
"""
import json
import os
import pickle
import time
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
import sitio as SI

ROOT = Path(__file__).parent
SITE = ROOT / "site"
CACHE = ROOT / "results/web_modelo.pkl"
PROG = Path.home() / ".claude/progress/web-modelo.json"
U = 0.4


def progreso(i, n):
    tmp = PROG.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": "web-modelo", "done": i, "total": n, "ts": time.time()}))
    os.replace(tmp, PROG)
    print(f"[{i}/{n}]", flush=True)


def entrenar():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    F0, g = E.armar()
    o, _ = M.cargar()
    for c in SI.ROS_COLS:
        F0[c] = F0[c].shift(24 * 7)
    tide = PR2.marea_feats(F0, o)
    cols = [c for k, v in g.items() if k != "rio_24h_antes" for c in v] + tide
    F0 = F0.dropna(subset=["y", "marea"])
    idx = F0.index
    S = PR.fuentes(idx)
    tr = F0[idx < PR.CORTE]
    calm = idx >= PR.CORTE
    base = dict(n_estimators=300, learning_rate=0.1, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    Xtr = imp.transform(tr[cols])
    reg = lgb.LGBMRegressor(**base, linear_tree=True, linear_lambda=1.0).fit(Xtr, tr.y); progreso(1, 11)
    cla = lgb.LGBMClassifier(**base).fit(Xtr, (tr.y < U).astype(int)); progreso(2, 11)
    grande = pickle.load(open(SI.CACHE, "rb"))
    y = F0.y[calm]
    clim = y.index.month.map(tr.groupby(tr.index.month).y.apply(lambda s: (s < U).mean())).values
    yb = (y < U).values
    cal, comp = {}, []
    for d in range(8):
        F = F0 if d == 0 else PR.armar_lead(F0, S, d, idx)
        X = imp.transform(F.loc[calm, cols])
        p, q = reg.predict(X), cla.predict_proba(X)[:, 1]
        niv = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        pro = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q, yb.astype(float))
        cal[d] = (niv, pro, float(np.std(y - niv.predict(p))))
        Xg = grande["imp"].transform(F.loc[calm, grande["cols"]])
        pg, qg = grande["reg"].predict(Xg), grande["cla"].predict_proba(Xg)[:, 1]
        bss = lambda pb: 1 - np.mean((pb - yb) ** 2) / np.mean((clim - yb) ** 2)
        comp.append(dict(lead=d, R2_web=1 - np.var(y - p) / np.var(y), R2_grande=1 - np.var(y - pg) / np.var(y),
                         BSS_web=bss(pro.predict(q)), BSS_grande=bss(grande["cal"][d][1].predict(qg))))
        print(pd.DataFrame(comp).round(3).tail(1).to_string(index=False), flush=True)
        progreso(3 + d, 11)
    m = dict(imp=imp, reg=reg, cla=cla, cal=cal, cols=cols, comp=comp)
    pickle.dump(m, open(CACHE, "wb"))
    return m


def arboles(booster):
    d = booster.dump_model()
    out = []
    for t in d["tree_info"]:
        f, th, le, ri, lv, lc, lfe, lco = [], [], [], [], [], [], [], []

        def nodo(n):
            if "leaf_index" in n or "leaf_value" in n and "split_index" not in n:
                k = len(lv)
                lv.append(n["leaf_value"])
                lc.append(n.get("leaf_const", n["leaf_value"]))
                lfe.append(n.get("leaf_features", []))
                lco.append(n.get("leaf_coeff", []))
                return -(k + 1)
            assert n["decision_type"] == "<=", n["decision_type"]
            k = len(f)
            f.append(n["split_feature"]); th.append(n["threshold"]); le.append(None); ri.append(None)
            le[k] = nodo(n["left_child"])
            ri[k] = nodo(n["right_child"])
            return k

        nodo(t["tree_structure"])
        arbol = dict(f=f, t=th, l=le, r=ri, v=lv)
        if any(lfe):
            arbol.update(c=lc, lf=lfe, lk=lco)
        out.append(arbol)
    return out


def riesgo(coef):
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
        tabla[str(thr)] = {("nino" if n else "normal"): [None if np.isnan(x := float(t.get((n, k), np.nan))) else round(x, 4)
                                                         for k in range(len(C.BINS) - 1)] for n in [False, True]}
    ros = w.ros.dropna()
    hn = h[h.nino].copy()
    sd_ = pd.DatetimeIndex(hn.salida).floor("D")
    ajuste = {}
    for estado in ["alto", "normal", "bajo"]:
        ajuste[estado] = {}
        for thr in C.UMBRALES:
            bajo = (hn.obs_min < thr).values
            fila = []
            for a, b in C.BANDAS:
                prev = ros.reindex(sd_ - pd.Timedelta(days=(a + b) // 2)).values
                st = np.where(prev > C.ROS_ALTO, "alto", np.where(prev < C.ROS_BAJO, "bajo", "normal"))
                k = st == estado
                fila.append(round(float(bajo[k].mean() / bajo.mean()), 3) if k.sum() > 300 else 1.0)
            ajuste[estado][str(thr)] = fila
    oni_ult = oni.dropna()
    return dict(bins=C.BINS, tabla=tabla, bandas=C.BANDAS, ajuste_rosario=ajuste, ros_alto=C.ROS_ALTO, ros_bajo=C.ROS_BAJO,
                ros_hist=[round(float(x), 2) for x in np.quantile(ros.values, np.linspace(0, 1, 101))],
                oni=dict(valor=float(oni_ult.iloc[-1]), mes=str(oni_ult.index[-1].date())))


def iso(m):
    return dict(x=[round(float(v), 6) for v in m.X_thresholds_], y=[round(float(v), 6) for v in m.y_thresholds_])


def main():
    m = entrenar()
    progreso(10, 11)
    o, _ = M.cargar()
    tr0 = o[o.index < M.CORTE].dropna()
    coef_vieja, *_ = np.linalg.lstsq(M.base_armonica(tr0.index), tr0.values, rcond=None)
    coef_nueva, *_ = np.linalg.lstsq(P.base_marea(tr0.index), tr0.values, rcond=None)
    ob = o.dropna()
    coef_todo, *_ = np.linalg.lstsq(P.base_marea(ob.index), ob.values, rcond=None)
    orden = [(float(p), bool(p < 30 and k not in ("S2", "S1", "T2", "P1"))) for k, p in {**M.PERIODOS, **P.EXTRA}.items()]
    puntos = {}
    for p in SI.PUNTOS:
        lat, lon = pd.read_csv(ROOT / f"data/era5/{p}.csv", nrows=1)[["latitude", "longitude"]].iloc[0]
        puntos[p] = [float(lat), float(lon)]
    modelo = dict(
        cols=m["cols"], medianas=[float(x) for x in m["imp"].statistics_],
        reg=arboles(m["reg"].booster_), cla=arboles(m["cla"].booster_),
        cal=[dict(nivel=iso(m["cal"][d][0]), prob=iso(m["cal"][d][1]), sd=m["cal"][d][2]) for d in range(8)],
        marea_vieja=dict(periodos=list(M.PERIODOS.values()), coef=coef_vieja.tolist()),
        marea_nueva=dict(periodos=orden, coef=coef_nueva.tolist(), nodal_h=P.NODAL),
        marea_display=dict(periodos=orden, coef=coef_todo.tolist(), nodal_h=P.NODAL),
        fases={c: M.PERIODOS[c] for c in P.FASES},
        puntos=puntos, puntos_sur=E.SUR, puntos_rdp=E.PUNTOS, lags=E.LAGS, umbral=U,
        riesgo=riesgo(coef_todo), feriados=SI.FERIADOS,
        comp=m["comp"], generado=pd.Timestamp.now(tz="UTC").isoformat(timespec="minutes"),
    )
    SITE.mkdir(exist_ok=True)
    (SITE / "model.json").write_text(json.dumps(modelo, separators=(",", ":")))
    print("model.json", (SITE / "model.json").stat().st_size // 1024, "KB", flush=True)

    # vector de prueba: features y predicciones de Python con el pronóstico de ahora
    fc = SI.bajar_pronostico()
    ahora = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    fin = min(f.index.max() for f in fc.values())
    era5_orig, cargar_orig = E.era5, M.cargar
    E.era5 = lambda p, idx: fc[p][["u", "v", "vel", "pres"]].reindex(idx).combine_first(era5_orig(p, idx))

    def cargar_ext():
        o_, w_ = cargar_orig()
        idx_ = pd.date_range(o_.index[0], fin, freq="h")
        return o_.reindex(idx_), w_.reindex(idx_).ffill(limit=24 * 10)

    M.cargar = cargar_ext
    try:
        F, g = E.armar()
        o2, _ = M.cargar()
    finally:
        E.era5, M.cargar = era5_orig, cargar_orig
    for c in SI.ROS_COLS:
        F[c] = F[c].shift(24 * 7)
    PR2.marea_feats(F, o2)
    F = F[(F.index >= ahora - pd.Timedelta(hours=24)) & (F.index <= fin - pd.Timedelta(hours=12))]
    X = m["imp"].transform(F[m["cols"]])
    ep = lambda ts: int(pd.Timestamp(ts).timestamp() // 3600)
    ros = json.load(open(ROOT / "data/rosario.json"))
    con = json.load(open(ROOT / "data/concordia.json"))
    corte = (ahora - pd.Timedelta(days=90)).isoformat()
    vec = dict(
        ahora=ep(ahora), horas=[ep(t) for t in F.index],
        forecast={p: dict(h0=ep(fc[p].index[0]), vel=fc[p].vel.round(4).tolist(), dir=fc[p]["dir"].tolist(), pres=fc[p].pres.tolist()) for p in SI.PUNTOS},
        rosario=[[o_["timestart"], o_["valor"]] for o_ in ros if o_["timestart"] >= corte],
        concordia=[[o_["timestart"], o_["valor"]] for o_ in con if o_["timestart"] >= corte],
        features=[[None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in row] for row in F[m["cols"]].values[::12]],
        feat_horas=[ep(t) for t in F.index[::12]],
        reg=m["reg"].predict(X).tolist(), cla=m["cla"].predict_proba(X)[:, 1].tolist(),
    )
    (ROOT / "results/test_vector.json").write_text(json.dumps(vec))
    print("test_vector ok", len(vec["horas"]), "horas", flush=True)
    progreso(11, 11)


if __name__ == "__main__":
    main()
