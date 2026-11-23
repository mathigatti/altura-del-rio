"""Calendario de riesgo de agua baja para la lancha Ramal 3 (Tigre ↔ Paraná de las Palmas por arroyo Caraguatá).

Riesgo = frecuencia histórica (2011-2026) de que el nivel mínimo en San Fernando durante el tramo del viaje
quede por debajo del umbral, condicionada a la marea astronómica mínima en ese tramo y a la fase ENSO.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import modelo as M

ART = "America/Argentina/Buenos_Aires"
VIAJES = [("ida", "08:00"), ("ida", "10:30"), ("ida", "14:00"), ("ida", "15:30"), ("ida", "18:30"),
          ("vuelta", "05:30"), ("vuelta", "11:00"), ("vuelta", "13:30"), ("vuelta", "17:00"), ("vuelta", "18:30")]
VENTANA = (30, 60)  # ida: entra al Caraguatá a los 30 min y llega al muelle a los 60; vuelta: muelle a los 30, sale del arroyo a los 60
PASO_MUELLE = {"ida": 60, "vuelta": 30}
UMBRALES = [0.3, 0.4, 0.5]
ROS_ALTO, ROS_BAJO = 4.3, 2.0  # m en Rosario (deciles aprox.)
BANDAS = [(0, 30), (30, 60), (60, 90), (90, 400)]
BINS = [-2, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 5]


def marea_todo(o, hasta):
    tr = o.dropna()
    coef, *_ = np.linalg.lstsq(M.base_armonica(tr.index), tr.values, rcond=None)
    idx = pd.date_range(o.index[0], hasta, freq="h")
    return pd.Series(M.base_armonica(idx) @ coef, index=idx)


def ventana_horas(salida_utc):
    a = salida_utc + pd.Timedelta(minutes=VENTANA[0])
    b = salida_utc + pd.Timedelta(minutes=VENTANA[1])
    return pd.date_range(a.ceil("h") if a.ceil("h") <= b else a.floor("h"), b.floor("h") if b.floor("h") >= a else b.ceil("h"), freq="h")


def viajes(fechas):
    filas = []
    for d in fechas:
        for sentido, hhmm in VIAJES:
            s = pd.Timestamp(f"{d} {hhmm}", tz=ART).tz_convert("UTC").tz_localize(None)
            filas.append(dict(fecha=d, sentido=sentido, hora=hhmm, salida=s))
    return pd.DataFrame(filas)


def oni_serie():
    oni = pd.read_csv(M.ROOT / "data/oni.txt", sep=r"\s+")
    mm = dict(zip(["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"], range(1, 13)))
    oni["t"] = pd.to_datetime(dict(year=oni.YR, month=oni.SEAS.map(mm), day=1))
    return oni.set_index("t").ANOM


def main():
    hasta = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2027-03-31")
    o, w = M.cargar()
    ros = w.ros.dropna()
    marea = marea_todo(o, hasta + pd.Timedelta(days=2))
    oni = oni_serie()
    nino = lambda t: oni.reindex(t.to_period("M").to_timestamp()).values > 0.5

    # histórico: todos los días 2011-2026 con los mismos horarios
    dias = pd.date_range("2011-01-02", o.index[-1] - pd.Timedelta(days=1), freq="D").date
    h = viajes(dias)
    h["marea_min"] = [marea.reindex(ventana_horas(s)).min() for s in h.salida]
    h["obs_min"] = [o.reindex(ventana_horas(s)).min() for s in h.salida]
    h = h.dropna(subset=["obs_min"])
    h["nino"] = nino(pd.DatetimeIndex(h.salida))
    h["bin"] = pd.cut(h.marea_min, BINS)

    tabla = {}
    for thr in UMBRALES:
        h["bajo"] = h.obs_min < thr
        tabla[thr] = h.groupby(["nino", "bin"], observed=True).bajo.agg(["mean", "count"])
    print("Riesgo histórico P(mín < 0,4 m) por marea mínima en el tramo, normal / con El Niño:")
    t4 = tabla[0.4]["mean"].unstack(0)
    t4.columns = ["normal", "El Niño"]
    print((t4 * 100).round(1).to_string(), "\n")

    # futuro: fines de semana (+ feriados si se pasan con --feriados=YYYY-MM-DD,...)
    hoy = date.today()
    fer = [pd.Timestamp(x).date() for a in sys.argv[2:] for x in a.split(",") if a]
    fechas = sorted({d.date() for d in pd.date_range(hoy, hasta) if d.weekday() >= 5} | set(fer))
    f = viajes(fechas)
    f["marea_min"] = [marea.reindex(ventana_horas(s)).min() for s in f.salida]
    f["bin"] = pd.cut(f.marea_min, BINS)
    f["nino"] = True  # El Niño fuerte en curso (ONI 2,46 a sep-2026); revisar oni.txt si cambia

    # Ajuste por Paraná: cuánto cambia el riesgo (dentro de meses Niño) si Rosario estaba alto N días antes
    ros_hoy = ros.iloc[-1]
    estado = "alto" if ros_hoy > ROS_ALTO else ("bajo" if ros_hoy < ROS_BAJO else "normal")
    hn = h[h.nino].copy()
    hn["salida_d"] = pd.DatetimeIndex(hn.salida).floor("D")
    ajuste = {thr: {} for thr in UMBRALES}
    for a, b in BANDAS:
        prev = ros.reindex(hn.salida_d - pd.Timedelta(days=(a + b) // 2)).values
        st = np.where(prev > ROS_ALTO, "alto", np.where(prev < ROS_BAJO, "bajo", "normal"))
        for thr in UMBRALES:
            bajo = (hn.obs_min < thr).values
            m = st == estado
            ajuste[thr][(a, b)] = bajo[m].mean() / bajo.mean() if m.sum() > 300 else 1.0
    print(f"Rosario hoy {ros_hoy:.2f} m ({estado}). Factor de riesgo según anticipación (umbral 0,4):",
          {f"{a}-{b}d": round(v, 2) for (a, b), v in ajuste[0.4].items()}, "\n")
    lead = (pd.to_datetime(f.fecha) - pd.Timestamp(hoy)).dt.days
    banda = [next(((a, b) for a, b in BANDAS if a <= L < b), BANDAS[-1]) for L in lead]
    for thr in UMBRALES:
        f[f"riesgo_{thr}"] = [tabla[thr]["mean"].get((n, b), np.nan) * ajuste[thr][bd]
                              for n, b, bd in zip(f.nino, f["bin"], banda)]
    f.drop(columns=["salida", "bin", "nino"]).to_csv(M.ROOT / "results/calendario.csv", index=False)

    g = f.pivot_table(index="fecha", columns=["sentido", "hora"], values="riesgo_0.4", sort=False)
    g = g[[("ida", h_) for _, h_ in VIAJES[:5]] + [("vuelta", h_) for _, h_ in VIAJES[5:]]]
    g.columns = [f"{'I' if s == 'ida' else 'V'}{(pd.Timestamp('2000-01-01 ' + h_) + pd.Timedelta(minutes=PASO_MUELLE[s])):%H:%M}" for s, h_ in g.columns]
    g.index = [f"{['lu','ma','mi','ju','vi','sá','do'][d.weekday()]} {d:%d/%m}{'*' if d in fer else ''}" for d in g.index]
    print("Riesgo % de agua < 0,4 m en San Fernando en el tramo del Caraguatá. Columnas = hora en que la lancha pasa por el muelle (I = ida, V = vuelta):")
    print((g * 100).round(0).astype("Int64").to_string())


if __name__ == "__main__":
    main()
