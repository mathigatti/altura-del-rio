#!/usr/bin/env python3
"""Archiva pronósticos (INA y SHN) y observaciones de San Fernando en datos.db.

Uso:
  python3 archivar.py             # corrida normal (cron)
  python3 archivar.py --backfill  # baja observaciones desde 2006
"""
import html
import json
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB = Path(__file__).with_name("datos.db")
INA = "https://alerta.ina.gob.ar/a5"
SHN = "https://www.hidro.gov.ar/oceanografia/pronostico.asp"
ART = timezone(timedelta(hours=-3))


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "altura-del-rio/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def init(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS obs_sf (t TEXT PRIMARY KEY, valor REAL);
    CREATE TABLE IF NOT EXISTS prono_ina (
        cor_id INTEGER, forecast_date TEXT, t TEXT, qualifier TEXT, valor REAL, bajado TEXT,
        PRIMARY KEY (cor_id, t, qualifier));
    CREATE TABLE IF NOT EXISTS prono_shn (
        bajado TEXT, lugar TEXT, estado TEXT, t TEXT, altura REAL, aviso TEXT,
        PRIMARY KEY (lugar, t, bajado));
    """)


def obs(con, desde, hasta):
    url = f"{INA}/obs/puntual/series/52/observaciones?timestart={desde}&timeend={hasta}&format=json"
    rows = [(utc(o["timestart"]), o["valor"]) for o in json.loads(get(url))]
    con.executemany("INSERT OR REPLACE INTO obs_sf VALUES (?,?)", rows)
    return len(rows)


def prono_ina(con, ahora):
    d = json.loads(get(f"{INA}/sim/calibrados/432/corridas/last?series_id=26202&includeProno=true&format=json"))
    fd = utc(d["forecast_date"])
    rows = [(d["id"], fd, utc(p["timestart"]), p.get("qualifier") or "main", p["valor"], ahora)
            for s in d["series"] for p in s["pronosticos"]]
    con.executemany("INSERT OR IGNORE INTO prono_ina VALUES (?,?,?,?,?,?)", rows)
    return d["id"], fd, len(rows)


def prono_shn(con, ahora):
    txt = html.unescape(re.sub(r"<[^>]+>", "\n", get(SHN)))
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    rows, aviso, lugar = [], "", None
    for i, l in enumerate(lines):
        if l.startswith("Corrección a las alturas"):
            aviso = lines[i + 1]
        elif i + 1 < len(lines) and lines[i + 1] in ("PLEAMAR", "BAJAMAR") and not re.fullmatch(r"\d+/\d+/\d+", l):
            lugar = l
        elif l in ("PLEAMAR", "BAJAMAR") and lugar and i + 3 < len(lines):
            hora, alt, fecha = lines[i + 1:i + 4]
            t = datetime.strptime(f"{fecha} {hora}", "%d/%m/%Y %H:%M").replace(tzinfo=ART)
            rows.append((ahora, lugar, l, t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"), float(alt), aviso))
    con.executemany("INSERT OR IGNORE INTO prono_shn VALUES (?,?,?,?,?,?)", rows)
    return len(rows)


def main():
    con = sqlite3.connect(DB)
    init(con)
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    hoy = datetime.now(timezone.utc)
    if "--backfill" in sys.argv:
        for y in range(2006, hoy.year + 1):
            print(y, obs(con, f"{y}-01-01", f"{y + 1}-01-01"), flush=True)
            con.commit()
    for nombre, f in [("obs", lambda: obs(con, (hoy - timedelta(days=3)).strftime("%Y-%m-%d"),
                                          (hoy + timedelta(days=1)).strftime("%Y-%m-%d"))),
                      ("ina", lambda: prono_ina(con, ahora)),
                      ("shn", lambda: prono_shn(con, ahora))]:
        try:
            print(ahora, nombre, f(), flush=True)
            con.commit()
        except Exception as e:
            print(ahora, nombre, "ERROR", repr(e), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
