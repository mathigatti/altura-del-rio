# ¿Pasa la lancha por el Caraguatá?

Elegís una fecha y la página te dice si la lancha colectiva **Ramal 3** (Tigre → Paraná de las Palmas) va a poder entrar al **arroyo Caraguatá**, en el Delta del Paraná. Con el río bajo, la lancha va lenta o no pasa.

Todo corre en el navegador, con datos en vivo:

- **Días pasados:** nivel medido en San Fernando ([INA](https://alerta.ina.gob.ar), serie 52).
- **Próximos 7 días:** un modelo propio (LightGBM) usa el pronóstico de viento y presión ECMWF ([Open-Meteo](https://open-meteo.com)) en 10 puntos del Río de la Plata y la plataforma, la marea astronómica y los ríos Paraná (Rosario) y Uruguay (Concordia).
- **Más adelante:** frecuencia histórica 2011–2026 de agua baja según la marea astronómica del horario del viaje, El Niño y el estado del Paraná.

## Qué tan bien anda

Prueba con pronósticos de viento ECMWF reales archivados, en 2025–2026. No es viento "perfecto".

| Antelación | R² del nivel | Puntaje de P(< 0,4 m) (BSS) |
|---|---|---|
| 1 día | 0,86 | 0,53 |
| 3 días | 0,83 | 0,48 |
| 5 días | 0,71 | 0,41 |
| 7 días | 0,47 | 0,24 |

El R² sube a 0,89 con viento real medido (ERA5). El límite de los pronósticos es el pronóstico del viento.

## Estructura

| Archivo | Qué hace |
|---|---|
| `site/` | Página estática (Netlify): `index.html`, `modelo.js` (features + árboles + calibración), `model.json`. |
| `archivar.py` | Archiva observaciones de San Fernando y los pronósticos del INA y del SHN en SQLite (cron cada 3 h). |
| `modelo.py`, `explicar.py`, `familias.py`, `empujar.py` | Experimentos: marea armónica, ablación de variables, familias de modelos, marea con modulación nodal. |
| `pronos.py`, `pronos2.py` | Evaluación honesta con pronósticos ECMWF archivados (Open-Meteo Previous Runs) y calibración isotónica. |
| `calendario.py` | Riesgo de largo plazo por viaje (marea × El Niño × Paraná). |
| `web_modelo.py` | Entrena la versión liviana para la web y exporta `site/model.json` y un vector de prueba. |
| `test_modelo_js.js` | Verifica que `modelo.js` dé lo mismo que Python (`node test_modelo_js.js`). |
| `results/*.csv` | Métricas de cada experimento. |

Los datos crudos no están en el repo:
- observaciones del INA: `archivar.py --backfill`;
- ERA5 de Open-Meteo, en 10 puntos (coordenadas en `site/model.json` → `puntos`);
- pronósticos archivados: `pronos.py` los baja solo.

## Límites

- El umbral (~0,40 m en San Fernando) sale de pocas observaciones propias y es aproximado.
- El esquema del mapa no está a escala.
- Supone que El Niño dura hasta abril de 2027.
