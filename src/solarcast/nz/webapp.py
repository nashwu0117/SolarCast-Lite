"""SolarCast NZ web app: a Flask server with a Leaflet map of New Zealand.

The page is a single HTML string (INDEX_HTML below), so there is no build step and no static folder to find.
Routes:
    GET /              the single-page app
    GET /api/times     {"times": [ISO UTC strings], "default": ISO UTC string}
    GET /api/snapshot  ?t=<ISO UTC | live>             -> {"time": t, "regions": [...]}
    GET /api/model     -> Service.model_report(): periods, overall, per_region, importance, sample
    GET /api/detail    ?region=<name>&t=<ISO | live>   -> the dict from Service.detail (plus "region", "time")
    GET /api/point     ?lat=&lon=&t=                   -> Service.point: forecast for ANY clicked NZ location
    GET /api/chat/status                              -> SolarCast AI state (off / downloading / ready / templates)
    POST /api/chat     {question, t, region, history} -> {answer, mode} (see assistant.py; rate-limited)

The app only talks to a `service` object (see docs/NZ_CONTRACT.md, section nz/service.py):
    available_times(), default_time(), snapshot(t), detail(region, t)

`serve(service)` starts the app in a background thread and, optionally, opens a free Cloudflare quick tunnel so
that people outside the machine (for example hackathon judges) can open the page.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from flask import Flask, Response, request

from solarcast.nz import config, ev

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TUNNEL_TIMEOUT_S = 60


# ----------------------------------------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------------------------------------
def _clean(value):
    """Make a value JSON-safe: numpy numbers -> Python, NaN/inf -> None, tuples -> lists."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and numpy scalars
        return _clean(value.tolist())
    if hasattr(value, "isoformat"):  # datetime / Timestamp
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(payload, status=200) -> Response:
    return Response(json.dumps(_clean(payload)), status=status, mimetype="application/json")


# ----------------------------------------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------------------------------------
def create_app(service, assistant=None, grid=None) -> Flask:
    """Build the Flask app around a service object (optionally the SolarCast AI assistant and the nationwide grid)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024          # chat requests are small; refuse huge bodies

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/times")
    def api_times():
        window = service.replay_window() if hasattr(service, "replay_window") else "history"
        ahead = service.ahead_options() if hasattr(service, "ahead_options") else [{"hours": 1}]
        return _json({"times": list(service.available_times()), "default": service.default_time(),
                      "replay_window": window, "ahead": ahead,
                      "reference_kw": config.SYSTEM_KW, "grid_price_nzd": config.GRID_PRICE_NZD,
                      "ev_cars": ev.CARS, "home_batteries": ev.HOME_BATTERIES, "home_charger_kw": ev.HOME_CHARGER_KW,
                      "three_phase_charger_kw": ev.THREE_PHASE_CHARGER_KW})

    @app.get("/api/snapshot")
    def api_snapshot():
        t = request.args.get("t") or service.default_time()
        try:
            regions = service.snapshot(t)
        except (ValueError, KeyError) as exc:
            return _json({"error": f"bad time: {exc}"}, 400)
        except Exception as exc:  # e.g. live data download failed
            return _json({"error": f"snapshot failed: {exc}"}, 502)
        return _json({"time": t, "regions": regions})

    @app.get("/api/model")
    def api_model():
        try:
            return _json(service.model_report())
        except Exception as exc:
            return _json({"error": f"model report failed: {exc}"}, 502)

    @app.get("/api/detail")
    def api_detail():
        region = request.args.get("region", "")
        t = request.args.get("t") or service.default_time()
        if not region:
            return _json({"error": "missing region"}, 400)
        try:
            data = dict(service.detail(region, t))
        except KeyError:
            return _json({"error": f"unknown region: {region}"}, 404)
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        except Exception as exc:
            return _json({"error": f"detail failed: {exc}"}, 502)
        data.setdefault("region", region)
        data.setdefault("time", t)
        return _json(data)

    # ----- any clicked point (downloads that point's weather; rate-limited per visitor) -----
    point_hits: dict = {}

    @app.get("/api/point")
    def api_point():
        if not hasattr(service, "point"):
            return _json({"error": "clicking any point is not supported by this service"}, 404)
        try:
            lat, lon = float(request.args["lat"]), float(request.args["lon"])
        except (KeyError, ValueError):
            return _json({"error": "missing or bad lat/lon"}, 400)
        client = request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"
        now = time.time()
        hits = [h for h in point_hits.get(client, []) if now - h < 60]
        if len(hits) >= 20:
            return _json({"error": "Too many clicks; please wait a minute."}, 429)
        point_hits[client] = hits + [now]
        try:
            return _json(service.point(lat, lon, request.args.get("t") or service.default_time()))
        except ValueError as exc:
            return _json({"error": str(exc)}, 400)
        except Exception as exc:  # e.g. Open-Meteo unreachable or rate-limited
            return _json({"error": f"could not get this point's weather ({type(exc).__name__}); try again"}, 502)

    # ----- nationwide grid (precomputed; refreshed every hour at :50) -----
    @app.get("/api/grid")
    def api_grid():
        if grid is None:
            return _json({"error": "The nationwide grid is not enabled."}, 404)
        t = request.args.get("t") or service.default_time()
        try:
            return _json(grid.snapshot(t))
        except (ValueError, KeyError) as exc:
            return _json({"error": f"bad time: {exc}"}, 400)
        except Exception as exc:  # live download failed, or the replay-day budget is used up
            return _json({"error": str(exc) if isinstance(exc, RuntimeError) else
                          f"grid failed ({type(exc).__name__})"}, 502)

    # ----- SolarCast AI chat (optional; templates only when no language model is loaded) -----
    @app.get("/api/chat/status")
    def api_chat_status():
        if assistant is None:
            return _json({"state": "off", "detail": "The AI assistant is not enabled."})
        return _json(assistant.status())

    @app.post("/api/chat")
    def api_chat():
        if assistant is None:
            return _json({"error": "The AI assistant is not enabled."}, 404)
        body = request.get_json(silent=True) or {}
        client = request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"
        if not assistant.allow(client):
            return _json({"error": "Too many questions; please wait a minute."}, 429)
        history = body.get("history") if isinstance(body.get("history"), list) else []
        pt = body.get("point") if isinstance(body.get("point"), dict) else None
        try:
            point = (float(pt["lat"]), float(pt["lon"])) if pt else None
        except (KeyError, TypeError, ValueError):
            point = None
        try:
            system_kw = min(max(float(body.get("system_kw") or config.SYSTEM_KW), 0.5), 1000.0)
        except (TypeError, ValueError):
            system_kw = config.SYSTEM_KW
        try:
            res = assistant.ask(str(body.get("question", ""))[:2000], str(body.get("t") or "live"),
                                body.get("region") or None, history, point=point, system_kw=system_kw)
        except Exception as exc:
            return _json({"error": f"chat failed: {exc}"}, 502)
        return _json(res)

    # ----- EV charging from the user's own panels (pure calculation on the forecast the page already has) -----
    @app.post("/api/ev_plan")
    def api_ev_plan():
        body = request.get_json(silent=True) or {}
        try:
            hours = [int(h) for h in body["hours"]][:48]
            solar = [None if v is None else float(v) for v in body["solar_kw"]][:48]
            car = str(body.get("car", "tesla_model_y"))
            hb = body.get("home_battery") or None
            if car not in ev.CARS or len(hours) != len(solar) or (hb and hb not in ev.HOME_BATTERIES):
                raise ValueError
            res = ev.plan(hours, solar, int(body.get("next_hour", 0)), car,
                          float(body.get("start_pct", 20)), float(body.get("target_pct", 80)),
                          min(max(float(body.get("charger_kw", ev.HOME_CHARGER_KW)), 1.4), 22.0), home_battery=hb)
        except (KeyError, TypeError, ValueError):
            return _json({"error": "bad EV plan request"}, 400)
        return _json(res)

    return app


# ----------------------------------------------------------------------------------------------------------
# serve(): background Flask thread + Cloudflare quick tunnel
# ----------------------------------------------------------------------------------------------------------
_STATE = {"server": None, "thread": None, "port": None, "proc": None, "tunnel_url": None}
_LOCK = threading.Lock()


def _start_server(app, port: int):
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, name="solarcast-web", daemon=True)
    thread.start()
    return server, thread


def _find_cloudflared() -> str | None:
    """Return the path of cloudflared, downloading the linux-amd64 binary only if it is missing."""
    found = shutil.which("cloudflared")
    if found:
        return found
    target = Path(os.environ.get("SOLARCAST_CACHE", Path.home() / ".cache" / "solarcast")) / "cloudflared"
    if not target.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading cloudflared to {target} ...")
            tmp = target.with_suffix(".part")
            urllib.request.urlretrieve(CLOUDFLARED_URL, tmp)
            tmp.replace(target)
            target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception as exc:
            print(f"WARNING: could not download cloudflared: {exc}")
            return None
    return str(target)


def _start_tunnel(port: int) -> tuple[subprocess.Popen | None, str | None]:
    """Start `cloudflared tunnel` and read the public URL from its stderr (up to 60 s)."""
    binary = _find_cloudflared()
    if binary is None:
        return None, None
    try:
        proc = subprocess.Popen(
            [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
    except OSError as exc:  # e.g. not a linux-amd64 machine
        print(f"WARNING: could not start cloudflared: {exc}")
        return None, None

    found: dict = {}
    got_url = threading.Event()

    def read_stderr():  # keeps draining stderr so cloudflared never blocks on a full pipe
        for line in proc.stderr:
            match = TUNNEL_URL_RE.search(line)
            if match and "url" not in found:
                found["url"] = match.group(0)
                got_url.set()

    threading.Thread(target=read_stderr, name="cloudflared-log", daemon=True).start()
    got_url.wait(TUNNEL_TIMEOUT_S)
    if "url" not in found:
        proc.terminate()
        return None, None
    return proc, found["url"]


def serve(service, port: int = 8000, tunnel: bool = True, assistant=None, grid=None) -> str:
    """Run the web app in a background thread; return the public tunnel URL (or the local URL).

    Safe to call twice: a second call reuses the running server and tunnel and returns the same URL.
    """
    with _LOCK:
        local_url = f"http://127.0.0.1:{port}"
        if _STATE["server"] is None:
            _STATE["server"], _STATE["thread"] = _start_server(create_app(service, assistant, grid), port)
            _STATE["port"] = port
            time.sleep(0.3)
        local_url = f"http://127.0.0.1:{_STATE['port']}"

        if not tunnel:
            return local_url
        if _STATE["proc"] is not None and _STATE["proc"].poll() is None and _STATE["tunnel_url"]:
            return _STATE["tunnel_url"]

        proc, url = _start_tunnel(_STATE["port"])
        if url is None:
            print(f"WARNING: no Cloudflare tunnel; the app is only reachable locally at {local_url}")
            return local_url
        _STATE["proc"], _STATE["tunnel_url"] = proc, url
        print(f"SolarCast NZ is live at {url}")
        return url


# ----------------------------------------------------------------------------------------------------------
# The page (HTML + CSS + JavaScript in one string; Leaflet and Chart.js come from CDNs with pinned versions)
# ----------------------------------------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SolarCast NZ</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f5f7fa; --card: #ffffff; --ink: #1c2733; --muted: #5d6b7a; --line: #dfe5ec;
    --accent: #e8890c; --good: #1f8f4d; --warn: #b26a00; --night: #243b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--ink); }
  header { padding: 14px 16px 10px; background: var(--card); border-bottom: 1px solid var(--line); }
  .top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; }
  h1 { margin: 0; font-size: 22px; letter-spacing: -0.3px; }
  h1 span { color: var(--accent); }
  .tagline { color: var(--muted); font-size: 13px; }
  .clock { text-align: right; }
  .clock .nz { font-size: 26px; font-weight: 700; line-height: 1.1; }
  .clock .nz small { font-size: 13px; font-weight: 500; color: var(--muted); }
  .clock .mode { font-size: 12px; color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px; margin-top: 10px; }
  .controls label { font-size: 13px; color: var(--muted); }
  #sizeSel, #sizeNum { font: inherit; font-size: 13px; padding: 4px 8px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
  #sizeNum { width: 76px; }
  .yours { background: #fff8e8; border: 1px solid #f3d9a4; border-radius: 12px; padding: 10px 12px; margin: 8px 0 4px; }
  .yours .big { font-size: 28px; font-weight: 700; color: #b45309; line-height: 1.1; }
  .yours .big small { font-size: 14px; font-weight: 500; color: var(--muted); }
  #slider { flex: 1 1 160px; min-width: 110px; accent-color: var(--accent); }
  header { background: linear-gradient(180deg, #fffaf0, #ffffff 70%); }
  .brand { display: flex; align-items: center; gap: 10px; }
  .logo { width: 38px; height: 38px; flex: none; animation: spin 40s linear infinite; }
  .seg { display: inline-flex; background: var(--bg); border: 1px solid var(--line); border-radius: 999px; padding: 3px; gap: 2px; }
  .seg button { border: 0; background: transparent; padding: 6px 14px; }
  .seg button.on { background: var(--ink); color: #fff; font-weight: 600; }
  .seg #liveBtn.on { background: #d14343; }
  .dot-live { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: currentColor; margin-right: 6px; vertical-align: 1px; }
  #liveBtn.on .dot-live { animation: blink 1.4s ease-in-out infinite; }
  .replay { display: flex; align-items: center; gap: 8px; flex: 1 1 440px; background: #fff8ec; border: 1px solid #f3d9a4; border-radius: 999px; padding: 4px 6px 4px 10px; }
  .replay[hidden] { display: none; }
  .predict { background: #eef4ff; border-color: #bcd0f5; flex-wrap: wrap; border-radius: 16px; }
  .predict .plabel { font-size: 13px; color: var(--muted); }
  .predict .ah { padding: 4px 12px; border-radius: 999px; }
  .predict .ah.on { background: #2563eb; border-color: #2563eb; color: #fff; font-weight: 600; }
  .predict .anote { font-size: 12px; color: var(--muted); flex: 1 1 220px; }
  .seg #predictBtn.on { background: #2563eb; }
  .badge.predict { background: #2563eb; }
  .replay input[type=date] { font: inherit; font-size: 13px; border: 1px solid var(--line); border-radius: 8px; padding: 3px 6px; background: #fff; }
  .replay .icon { padding: 4px 10px; }
  .replay .play.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  #rHour { font-variant-numeric: tabular-nums; min-width: 44px; }
  .sizebox { display: flex; align-items: center; gap: 6px; margin-left: auto; }
  .badge { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 1px 8px; font-size: 11px; font-weight: 700; letter-spacing: .5px; color: #fff; vertical-align: 1px; }
  .badge.live { background: #d14343; }
  .badge.live i { width: 7px; height: 7px; border-radius: 50%; background: #fff; animation: blink 1.4s ease-in-out infinite; }
  .badge.replay { background: var(--accent); }
  @keyframes blink { 50% { opacity: .25; } }
  .note button { padding: 2px 12px; margin: 0 6px; font-size: 13px; }
  @media (max-width: 900px) { .sizebox { margin-left: 0; } .replay { border-radius: 12px; flex-wrap: wrap; } }
  @media (prefers-reduced-motion: reduce) { .logo, .dot-live, .badge.live i { animation: none !important; } }
  button { font: inherit; border: 1px solid var(--line); background: var(--card); color: var(--ink); padding: 6px 14px; border-radius: 999px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .note { margin: 10px 0 0; padding: 8px 12px; border-radius: 8px; background: #fff6e5; color: #6b4a00; font-size: 13px; }
  .note.night { background: #e8edf8; color: var(--night); }
  .note.error { background: #fdeaea; color: #8a1f1f; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 14px; padding: 14px 16px; align-items: start; }
  .mapwrap { position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
  #map { height: 72vh; min-height: 420px; }
  .legend { position: absolute; left: 10px; bottom: 10px; z-index: 500; background: rgba(255,255,255,.94); border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; font-size: 12px; }
  .legend b { display: block; margin-bottom: 4px; }
  .legend div { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  .dot { width: 12px; height: 12px; border-radius: 50%; border: 1px solid rgba(0,0,0,.35); flex: none; }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .panel h2 { margin: 0 0 2px; font-size: 19px; }
  .panel h3 { margin: 16px 0 6px; font-size: 14px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .4px; }
  .empty { color: var(--muted); padding: 24px 4px; text-align: center; }
  .sub { color: var(--muted); font-size: 13px; }
  .chartbox { position: relative; height: 250px; }
  .advice { border-radius: 10px; padding: 10px 12px; border: 1px solid; }
  .advice.yes { background: #e9f7ee; border-color: #b6e2c6; }
  .advice.no { background: #fff4e0; border-color: #f0d6a6; }
  .advice strong { font-size: 16px; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .stat { background: var(--bg); border-radius: 10px; padding: 8px 6px; text-align: center; }
  .stat b { display: block; font-size: 18px; }
  .stat span { font-size: 12px; color: var(--muted); }
  table.m { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.m td { padding: 3px 0; border-bottom: 1px solid var(--line); }
  table.m td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .tabs { display: flex; gap: 6px; margin-top: 10px; }
  .tabs button.on { background: var(--ink); border-color: var(--ink); }
  .maptools { padding: 8px 12px; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--muted); display: flex; gap: 8px; align-items: center; }
  .maptools select { font: inherit; padding: 3px 6px; border-radius: 6px; border: 1px solid var(--line); }
  #modelView { padding: 14px 16px; display: none; }
  .grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  .grid2 .wide { grid-column: 1 / -1; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .card h3 { margin: 0 0 4px; font-size: 15px; }
  .card .chartbox { height: 280px; }
  .card .chartbox.tall { height: 360px; }
  table.m th { text-align: left; font-weight: 600; color: var(--muted); border-bottom: 1px solid var(--line); padding: 3px 0; }
  table.m th:not(:first-child) { text-align: right; }
  @media (max-width: 900px) { .grid2 { grid-template-columns: minmax(0, 1fr); } #modelView { padding: 10px 12px; } }
  footer { padding: 6px 16px 22px; color: var(--muted); font-size: 12px; }
  .leaflet-tooltip.tip { font-size: 13px; line-height: 1.4; padding: 8px 10px; border-radius: 8px; }
  .tip b.gen { color: var(--good); } .tip b.nogen { color: #6b7684; }
  .rname { background: none; border: 0; box-shadow: none; font-size: 11px; font-weight: 600; color: #26323f; text-shadow: 0 0 3px #fff, 0 0 3px #fff, 0 0 3px #fff; padding: 0; }
  .rname::before { display: none; }
  @media (max-width: 900px) {
    main { grid-template-columns: minmax(0, 1fr); padding: 10px 12px; }
    #map { height: 58vh; min-height: 340px; }
    .clock { text-align: left; }
    .chartbox { height: 230px; }
  }
  /* ----- nationwide grid and AI briefing ----- */
  .brief { padding: 10px 12px; border-bottom: 1px solid var(--line); background: linear-gradient(90deg, #fff8ec, #ffffff); }
  .brief .bhead { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  .brief .ai { background: var(--ink); color: #fff; border-radius: 999px; padding: 1px 9px; font-weight: 600; font-size: 11px; letter-spacing: .3px; }
  .brief .ai.tpl { background: #8a96a3; }
  #briefText { font-size: 14px; }
  #briefStats { font-size: 12px; color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums; }
  .brief.err #briefText { color: #8a1f1f; font-size: 13px; }
  .maptools .gt { margin-left: auto; display: flex; align-items: center; gap: 5px; cursor: pointer; }
  /* ----- SolarCast AI chat ----- */
  #chatBtn { position: fixed; right: 18px; bottom: 18px; z-index: 1001; background: var(--ink); color: #fff; border: 0;
             padding: 12px 18px; font-weight: 600; box-shadow: 0 6px 18px rgba(0,0,0,.25); display: none; }
  #chatBtn:hover { background: #2d3d4f; }
  #chat { position: fixed; right: 18px; bottom: 18px; z-index: 1002; width: 400px; max-width: calc(100vw - 32px);
          height: min(620px, calc(100vh - 36px)); background: var(--card); border: 1px solid var(--line); border-radius: 14px;
          box-shadow: 0 12px 34px rgba(0,0,0,.22); display: none; flex-direction: column; overflow: hidden; }
  #chat.open { display: flex; }
  .chead { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 8px; }
  .chead b { flex: 1; } .chead button { padding: 2px 10px; }
  #chatStatus { font-size: 12px; color: var(--muted); padding: 6px 12px; background: var(--bg); border-bottom: 1px solid var(--line); }
  #chatStatus.ready { color: var(--good); }
  #chatLog { flex: 1; overflow-y: auto; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px; }
  .msg { max-width: 88%; padding: 8px 11px; border-radius: 12px; font-size: 14px; white-space: normal; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: var(--accent); color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot { align-self: flex-start; background: var(--bg); border-bottom-left-radius: 4px; }
  .msg.bot.wait { color: var(--muted); font-style: italic; }
  .msg .tag { display: block; font-size: 11px; color: var(--muted); margin-top: 4px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 12px 8px; }
  .chips button { font-size: 12px; padding: 4px 10px; }
  #chatCtx { font-size: 11px; color: var(--muted); padding: 0 12px 6px; }
  .cform { display: flex; gap: 6px; padding: 8px 12px 12px; border-top: 1px solid var(--line); }
  .cform input { flex: 1; font: inherit; padding: 8px 10px; border: 1px solid var(--line); border-radius: 999px; min-width: 0; }
  .cform button { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .cform button:disabled { opacity: .5; cursor: default; }
  @media (max-width: 600px) { #chat { right: 8px; left: 8px; bottom: 8px; width: auto; max-width: none; height: calc(100vh - 16px); } }
  /* Power-flow animation: sun -> your panels -> home -> EV */
  .flow { display: block; width: 100%; max-width: 420px; height: auto; margin: 6px 0 2px; }
  .flow text { font: 600 10px system-ui, sans-serif; fill: var(--muted); }
  .flow .kw { fill: var(--good); font-size: 11px; }
  .flow .wire { fill: none; stroke: var(--line); stroke-width: 5; stroke-linecap: round; }
  .flow .juice { fill: none; stroke: var(--good); stroke-width: 3; stroke-linecap: round; stroke-dasharray: 4 12;
                 animation: flowdash var(--speed, 1.5s) linear infinite; }
  .flow .grid-juice { fill: none; stroke: var(--night); stroke-width: 3; stroke-linecap: round; stroke-dasharray: 4 12;
                      animation: flowdash 2.2s linear infinite; }
  .flow .sunrays { transform-origin: 30px 26px; animation: spin 14s linear infinite; }
  @keyframes flowdash { to { stroke-dashoffset: -32; } }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .flow .juice, .flow .grid-juice, .flow .sunrays { animation: none; } }
  /* Charge your EV from your panels */
  .evc { background: #effaf2; border: 1px solid #bfe3cb; border-radius: 12px; padding: 10px 12px; margin: 8px 0 4px; }
  .evcars { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .evcars { align-items: center; }
  .evcars input[type=search] { flex: 1; min-width: 0; font: inherit; padding: 6px 10px; border: 1px solid #bfe3cb; border-radius: 10px; background: #fff; }
  .evcars #evCount { flex-basis: 100%; margin: -2px 0 0 40px; }
  .evcars #evCount:empty { display: none; }
  .hb { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #bfe3cb; }
  .hb h4 { margin: 0 0 6px; font-size: 14px; }
  .hb select { width: 100%; font: inherit; font-weight: 600; padding: 6px 8px; border: 1px solid #bfe3cb; border-radius: 10px; background: #fff; }
  .hbb .b1 { background: repeating-linear-gradient(135deg, #f5b400 0 10px, #e09b00 10px 20px); background-size: 28px 28px; }
  .evcars select { flex: 1 1 100%; min-width: 0; font: inherit; font-weight: 600; padding: 6px 8px; border: 1px solid #bfe3cb; border-radius: 10px; background: #fff; }
  .evcars svg { width: 34px; height: 20px; flex: none; }
  .evin { display: grid; grid-template-columns: auto 1fr auto; gap: 4px 8px; align-items: center; font-size: 13px; color: var(--muted); }
  .evin input[type=range] { width: 100%; accent-color: var(--good); }
  .evin b { color: var(--ink); min-width: 38px; text-align: right; }
  .evin select { grid-column: 2 / 4; font: inherit; padding: 3px 6px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
  .batt { position: relative; height: 34px; border: 3px solid #1c2733; border-radius: 8px; margin: 12px 8px 6px 0; background: #fff; overflow: hidden; }
  .batt .b0 { position: absolute; left: 0; top: 0; bottom: 0; background: #cbd5e1; }
  .batt .b1 { position: absolute; top: 0; bottom: 0; width: 0; transition: width 1.4s cubic-bezier(.2,.8,.2,1);
              background: repeating-linear-gradient(135deg, #22a559 0 10px, #1f8f4d 10px 20px); background-size: 28px 28px;
              animation: charge 1s linear infinite; }
  .batt .tgt { position: absolute; top: -2px; bottom: -2px; width: 0; border-left: 2px dashed #1c2733; }
  .batt .lbl { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #1c2733; text-shadow: 0 0 3px #fff, 0 0 3px #fff; }
  @keyframes charge { to { background-position: 28px 0; } }
  .evhead { font-size: 15px; font-weight: 600; margin: 4px 0; }
  .evhead.ok { color: var(--good); } .evhead.part { color: var(--warn); }
  .evbars { display: block; width: 100%; height: 86px; margin-top: 4px; }
  .evbars rect.bar { transform-box: fill-box; transform-origin: bottom; animation: grow .7s cubic-bezier(.2,.8,.2,1) both; }
  .evbars text { font: 9px system-ui, sans-serif; fill: var(--muted); }
  @keyframes grow { from { transform: scaleY(0); } }
  .evkey { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11px; color: var(--muted); }
  .evkey i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  .leaflet-interactive.selpulse { animation: selpulse 1.8s ease-in-out infinite; }
  @keyframes selpulse { 50% { stroke-width: 7; stroke-opacity: .55; } }
  @media (prefers-reduced-motion: reduce) { .batt .b1, .evbars rect.bar, .leaflet-interactive.selpulse { animation: none; transition: none; } }
</style>
</head>
<body>
<header>
  <div class="top">
    <div class="brand">
      <svg class="logo" viewBox="0 0 40 40" aria-hidden="true"><g stroke="#f5a623" stroke-width="3" stroke-linecap="round"><path d="M20 3v5M20 32v5M3 20h5M32 20h5M8 8l3.5 3.5M28.5 28.5L32 32M8 32l3.5-3.5M28.5 11.5L32 8"/></g><circle cx="20" cy="20" r="8" fill="#f5b400"/></svg>
      <div><h1>SolarCast <span>NZ</span></h1>
      <div class="tagline">How much power your solar panels will give next hour, anywhere in New Zealand</div></div>
    </div>
    <div class="clock">
      <div class="nz" id="nzTime">--</div>
      <div class="mode"><span class="badge" id="modeBadge"></span> <span id="modeLabel">Loading...</span></div>
    </div>
  </div>
  <div class="controls">
    <div class="seg" role="group" aria-label="Which hour to show">
      <button id="replayBtn" type="button">&#9194; Replay past 24 h</button>
      <button id="liveBtn" type="button"><i class="dot-live"></i>Live now</button>
      <button id="predictBtn" type="button">&#128302; Predict next</button>
    </div>
    <div class="replay" id="replayBox" hidden>
      <button id="rPrev" type="button" class="icon" title="One hour back" aria-label="One hour back">&#9664;</button>
      <input type="range" id="slider" min="0" max="23" value="12" aria-label="Which of the past hours">
      <b id="rHour">--</b>
      <button id="rNext" type="button" class="icon" title="One hour forward" aria-label="One hour forward">&#9654;</button>
      <button id="rPlay" type="button" class="play">&#9654; Play</button>
    </div>
    <div class="replay predict" id="predictBox" hidden>
      <span class="plabel">How far ahead?</span><span id="aheadBtns"></span><span id="aheadNote" class="anote"></span>
    </div>
    <div class="sizebox">
    <label for="sizeSel">&#9728; Your panels</label>
    <select id="sizeSel" aria-label="Your solar system size">
      <option value="3">3 kW (small home)</option>
      <option value="5" selected>5 kW (typical home)</option>
      <option value="10">10 kW (large home / farm house)</option>
      <option value="50">50 kW (business / school)</option>
      <option value="100">100 kW (large roof)</option>
      <option value="custom">Other size (type it)</option>
    </select>
    <input id="sizeNum" type="number" min="0.5" max="1000" step="0.5" value="5" aria-label="Your solar system size in kW"><span class="tagline">kW</span>
    </div>
  </div>
  <div class="tabs">
    <button id="tabMap" type="button" class="on">Map</button>
    <button id="tabModel" type="button">Model</button>
  </div>
  <p class="note" id="note"></p>
</header>

<main id="mapView">
  <div class="mapwrap">
    <div class="brief" id="brief" hidden>
      <div class="bhead"><span class="ai" id="briefTag">AI briefing</span><span id="briefMeta"></span></div>
      <div id="briefText"></div>
      <div id="briefStats"></div>
    </div>
    <div class="maptools"><label for="colorMode">Colour circles by</label>
      <select id="colorMode"><option value="power">Predicted next-hour power (your panels)</option><option value="accuracy">How accurate the forecast is</option></select>
      <label class="gt" id="gridTool"><input type="checkbox" id="gridOn" checked> Nationwide grid</label></div>
    <div id="map"></div>
    <div class="legend" id="legend"></div>
  </div>
  <aside class="panel" id="panel">
    <div class="empty">Click a region circle, or <b>anywhere in New Zealand</b>, to see how much power your panels will give there: next hour, the hourly chart for the day, and an example use (charging an EV from solar).</div>
  </aside>
</main>

<section id="modelView">
  <div class="grid2">
    <div class="card wide"><h3>How good is our forecast?</h3><div class="sub" id="mPeriods">Loading...</div></div>
    <div class="card wide"><h3 id="hVersion">Model version</h3><div id="mVersion" class="sub">Loading...</div></div>
    <div class="card"><h3 id="hOverallKw">How far off, on average (kW)</h3><div class="sub">The average gap between the forecast and what the panels really made, in daytime. Shorter bar = better.</div><div class="chartbox"><canvas id="cOverallKw"></canvas></div></div>
    <div class="card"><h3 id="hOverallWm">How far off in sunlight (W/m&sup2;)</h3><div class="sub">The same gap measured in sunlight on each square metre, the unit scientists use. Shorter bar = better.</div><div class="chartbox"><canvas id="cOverallWm"></canvas></div></div>
    <div class="card wide"><h3>All the numbers</h3><div id="mTable"></div></div>
    <div class="card wide"><h3>How far off in each region (kW)</h3><div class="sub">Average gap in the test months, shorter bar = better. Our forecast wins in all 14 regions.</div><div class="chartbox tall"><canvas id="cRegion"></canvas></div></div>
    <div class="card wide"><h3 id="hSample">One example week</h3><div class="sub">What we forecast vs what the panels really made</div><div class="chartbox tall"><canvas id="cSample"></canvas></div></div>
    <div class="card wide"><h3>What the forecast pays most attention to</h3><div class="sub">Share of the model's decisions that use each piece of information (top 10)</div><div class="chartbox tall"><canvas id="cImp"></canvas></div></div>
  </div>
</section>

<button id="chatBtn" type="button" aria-controls="chat">Ask SolarCast AI</button>
<section id="chat" role="dialog" aria-label="SolarCast AI chat">
  <div class="chead"><b>SolarCast AI</b><button id="chatClear" type="button" title="Clear the chat">Clear</button><button id="chatClose" type="button" aria-label="Close">&times;</button></div>
  <div id="chatStatus">Checking the AI model...</div>
  <div id="chatLog" aria-live="polite">
    <div class="msg bot">Hi! Ask me how much power solar panels will give next hour anywhere in New Zealand (I use the panel size you picked), the weather, the EV charging example, or how accurate our model is. I answer only from this data and cannot browse the internet. 中文也可以問。</div>
  </div>
  <div class="chips" id="chatChips">
    <button type="button">Where in New Zealand is it sunniest next hour?</button>
    <button type="button">How much will my panels make next hour in Christchurch?</button>
    <button type="button">How accurate is the model?</button>
    <button type="button">How long to charge a Tesla Model Y from my panels today?</button>
    <button type="button">用我的太陽能板幫比亞迪 Atto 3 充電要多久？</button>
    <button type="button">我的太陽能板今天在奧克蘭能發多少電？</button>
  </div>
  <div id="chatCtx"></div>
  <form class="cform" id="chatForm"><input id="chatInput" maxlength="500" autocomplete="off" placeholder="Ask about this data..." aria-label="Your question"><button id="chatSend" type="submit">Send</button></form>
</section>

<footer>
  AI chat: Qwen3 (Apache-2.0) running locally on the notebook's machine, safety check by Llama Guard 3 (Built with Llama); it answers only from this dashboard's data and may make mistakes.
  Weather: Open-Meteo (CC BY 4.0), past weather and weather forecasts. Panel power is worked out from sunlight and scaled to the panel size you pick. Prototype; not energy or financial advice.
</footer>

<script>
// ---------- colours: predicted next-hour kW -> circle colour ----------
// The server forecasts a 50 kW reference system; output scales linearly with panel size, so every kW shown is
// multiplied by SIZE / REF. Colours use the share of rated power, which is the same for any size.
let SIZE = 5, REF = 50, PRICE = 0.30;
try { const v = +localStorage.getItem('solarcastSize'); if (v > 0) SIZE = v; } catch (e) {}
const S = v => (v === null || v === undefined || Number.isNaN(+v)) ? v : +v * SIZE / REF;
const kwd = () => SIZE < 20 ? 2 : 1;                        // decimals for kW at this size
const BINS = [
  { max: 0.02,     color: '#9aa5b1' },
  { max: 0.2,      color: '#fde68a' },
  { max: 0.4,      color: '#fbbf24' },
  { max: 0.6,      color: '#f97316' },
  { max: 0.8,      color: '#dc2626' },
  { max: Infinity, color: '#7f1d1d' },
];
function binLabel(i) {
  if (i === 0) return 'Not generating (night or heavy cloud)';
  const lo = BINS[i - 1].max * SIZE, hi = (BINS[i].max === Infinity ? 1 : BINS[i].max) * SIZE;
  return `${num(lo, lo < 10 ? 1 : 0)} - ${num(hi, hi < 10 ? 1 : 0)} kW`;
}
function colorFor(kw) { return (BINS.find(b => kw / REF < b.max) || BINS[BINS.length - 1]).color; }

// ---------- small helpers ----------
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(+v)) ? '--' : (+v).toFixed(d);
const NZ_FMT = new Intl.DateTimeFormat('en-NZ', { timeZone: 'Pacific/Auckland', weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
                                                  hour: '2-digit', minute: '2-digit', hour12: false });
function nzString(iso) {
  const d = new Date(iso);
  return isNaN(d) ? '--' : NZ_FMT.format(d);
}

let times = [], current = 'live', selected = null, markers = {}, chart = null, reqId = 0, detailId = 0, debounce = null;

// ---------- map ----------
const map = L.map('map', { zoomSnap: 0.25, scrollWheelZoom: true });
map.fitBounds([[-47.4, 166.0], [-34.2, 178.8]]);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 12,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

function tooltipHtml(r) {
  const gen = r.generating
    ? `<b class="gen">Your ${SIZE} kW panels: ${num(S(r.predicted_kw_next), kwd())} kW expected ${whenText()}</b>`
    : `<b class="nogen">Not generating${r.is_day ? ' (too cloudy)' : ' (night)'}: ${num(S(r.predicted_kw_next), kwd())} kW expected</b>`;
  const acc = (regionMae[r.name] !== undefined && $('colorMode').value === 'accuracy')
    ? `Forecast is off by ${num(S(regionMae[r.name]), 2)} kW on average here (your ${SIZE} kW system)<br>` : '';
  return `<b>${esc(r.name)}</b> &middot; ${esc(r.local_time)}<br>${acc}${gen}<br>` +
    `Cloud ${num(r.cloud_cover_pct, 0)}% &middot; ${num(r.temp_c)} &deg;C &middot; wind ${num(r.wind_kmh, 0)} km/h<br>` +
    `Sunlight now ${num(r.ghi_now_wm2, 0)} W/m&sup2; &middot; ${r.is_day ? 'daytime' : 'night'}`;
}

// Model accuracy colouring: bins are computed from the per-region MAE of our model (from /api/model).
const ACC_COLORS = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027'];
let report = null, regionMae = {}, accBins = [], lastRegions = [];
function ourModel(rows) {   // our model = any model name that is not a persistence baseline
  const names = [...new Set(rows.map(r => r.model))];
  return names.find(n => !/persist/i.test(n)) || names[0];
}
function prepareAccuracy() {
  regionMae = {}; accBins = [];
  if (!report || !(report.per_region || []).length) return;
  const mine = ourModel(report.per_region);
  report.per_region.filter(r => r.model === mine).forEach(r => { regionMae[r.region] = r.mae_kw; });
  const v = Object.values(regionMae).filter(x => typeof x === 'number');
  if (!v.length) return;
  const lo = Math.min(...v), hi = Math.max(...v), step = (hi - lo) / ACC_COLORS.length || 1;
  accBins = ACC_COLORS.map((c, i) => ({ color: c, from: lo + i * step, to: lo + (i + 1) * step }));
}
function accColor(name) {
  const v = regionMae[name];
  if (v === undefined || !accBins.length) return '#9aa5b1';
  const b = accBins.find(b => v <= b.to) || accBins[accBins.length - 1];
  return b.color;
}
function fillFor(r) { return $('colorMode').value === 'accuracy' ? accColor(r.name) : colorFor(r.predicted_kw_next); }
function drawLegend() {
  if ($('colorMode').value === 'accuracy') {
    $('legend').innerHTML = `<b>How far off the forecast is here, on average (kW, ${SIZE} kW system)</b>` + (accBins.length
      ? accBins.map(b => `<div><span class="dot" style="background:${b.color}"></span>${num(S(b.from), 2)} - ${num(S(b.to), 2)}</div>`).join('')
      : '<div>Model report not loaded</div>');
  } else {
    $('legend').innerHTML = `<b>Your ${SIZE} kW panels, next hour</b>` +
      BINS.map((b, i) => `<div><span class="dot" style="background:${b.color}"></span>${binLabel(i)}</div>`).join('');
  }
}
$('colorMode').onchange = () => { drawLegend(); drawRegions(lastRegions); };

function drawRegions(regions) {
  lastRegions = regions;
  regions.forEach(r => {
    const style = {
      radius: selected === r.name ? 17 : 14, color: selected === r.name ? '#1c2733' : '#ffffff',
      weight: selected === r.name ? 3 : 2, fillColor: fillFor(r), fillOpacity: 0.92,
    };
    let m = markers[r.name];
    if (!m) {
      m = L.circleMarker([r.lat, r.lon], { ...style, bubblingMouseEvents: false }).addTo(map);
      m.bindTooltip('', { sticky: true, direction: 'top', offset: [0, -8], className: 'tip' });
      m.on('click', () => selectRegion(r.name));
      m.label = L.marker([r.lat, r.lon], { interactive: false, icon: L.divIcon({ className: '', html: '', iconSize: [0, 0] }) })
        .bindTooltip(esc(r.name), { permanent: true, direction: 'bottom', offset: [0, 12], className: 'rname' }).addTo(map);
      markers[r.name] = m;
    }
    m.setStyle(style);
    m.setTooltipContent(tooltipHtml(r));
  });
}

// ---------- nationwide grid (about 120 land cells, precomputed on the server, refreshed at :50 every hour) ----------
map.createPane('gridPane'); map.getPane('gridPane').style.zIndex = 350;      // under the region circles
const gridRenderer = L.svg({ pane: 'gridPane' });
const NZ_HM = new Intl.DateTimeFormat('en-NZ', { timeZone: 'Pacific/Auckland', hour: '2-digit', minute: '2-digit', hour12: false });
const hm = iso => { const d = new Date(iso); return iso && !isNaN(d) ? NZ_HM.format(d) : '--'; };
let gridLayer = null, gridReq = 0, gridTimer = null, gridAvailable = true, lastGrid = null;

function drawGrid(g) {
  if (gridLayer) { map.removeLayer(gridLayer); gridLayer = null; }
  if (!g || !$('gridOn').checked) return;
  const h = g.step / 2;
  gridLayer = L.layerGroup((g.cells || []).map(c => {
    const on = c.kw / REF >= 0.02;
    return L.rectangle([[c.lat - h, c.lon - h], [c.lat + h, c.lon + h]], { renderer: gridRenderer, color: '#ffffff', weight: 0.6,
        opacity: 0.7, fillColor: colorFor(c.kw), fillOpacity: on ? 0.55 : 0.2 })
      .bindTooltip(`<b>Around ${esc(c.near)}</b> &middot; cell ${num(c.lat, 2)}, ${num(c.lon, 2)}<br>` +
        `<b class="${on ? 'gen' : 'nogen'}">Your ${SIZE} kW panels: ${num(S(c.kw), kwd())} kW next hour</b>` +
        (c.actual !== null && c.actual !== undefined ? `<br>Actual ${num(S(c.actual), kwd())} kW (derived from weather data)` : '') +
        `<br>Cloud ${num(c.cloud, 0)}% &middot; click for this spot's hourly detail`, { sticky: true, className: 'tip' });
  })).addTo(map);
}

function renderBrief(g) {
  const s = g.stats || {}, b = g.briefing || {}, el = $('brief');
  el.hidden = false; el.classList.remove('err');
  const ai = b.mode === 'llm';
  $('briefTag').textContent = ai ? 'AI briefing' : 'Briefing';
  $('briefTag').className = 'ai' + (ai ? '' : ' tpl');
  $('briefMeta').textContent = ai ? `written by ${b.model || 'the local AI model'} from the grid numbers below`
    : g.briefing_pending ? 'the AI model is writing a briefing...' : 'fixed template (the AI model is not ready)';
  $('briefText').textContent = b.text || '';
  const parts = [`${g.local_time} NZ, next hour`, `${s.cells || 0} cells`,
                 `your ${SIZE} kW panels: NZ average ${num(S(s.mean_kw), kwd())} kW, best ${num(S(s.max_kw), kwd())} kW`,
                 `${s.share_generating_pct ?? '--'}% generating`];
  if (s.mean_abs_error_kw !== undefined) parts.push(`grid error this hour ${num(S(s.mean_abs_error_kw), 2)} kW`);
  if (g.live) parts.push(`computed ${hm(g.updated_at)}, next update ${hm(g.next_update)} (every hour at :${g.refresh_minute})`);
  $('briefStats').textContent = parts.join(' \u00b7 ');
}

async function loadGrid() {
  if (!gridAvailable) return;
  const id = ++gridReq;
  clearTimeout(gridTimer);
  try {
    const g = await getJson('/api/grid?t=' + encodeURIComponent(current));
    if (id !== gridReq) return;
    const hourChanged = lastGrid && lastGrid.live && g.live && lastGrid.time !== g.time;
    lastGrid = g; drawGrid(g); renderBrief(g);
    if (g.briefing_pending) gridTimer = setTimeout(loadGrid, 4000);
    if (hourChanged) { if (selected) loadDetail(selected); else if (selectedPoint) loadPoint(); }
  } catch (e) {
    if (id !== gridReq) return;
    if (/not enabled/.test(e.message)) { gridAvailable = false; $('gridTool').style.display = 'none'; $('brief').hidden = true; return; }
    lastGrid = null; drawGrid(null);
    $('brief').hidden = false; $('brief').classList.add('err');
    $('briefTag').textContent = 'Nationwide grid'; $('briefTag').className = 'ai tpl'; $('briefMeta').textContent = '';
    $('briefText').textContent = e.message; $('briefStats').textContent = '';
    if (/still being computed/.test(e.message)) gridTimer = setTimeout(loadGrid, 10000);
  }
}
$('gridOn').onchange = () => drawGrid(lastGrid);

// Live mode refreshes itself: the map is checked every minute and switches when the server's new hour is ready.
setInterval(() => {
  if (!isReplay() && document.visibilityState === 'visible') loadSnapshot({ panel: false });
}, 60000);
setInterval(() => { if (!isReplay()) headerClock(); }, 15000);       // keep the Live clock ticking

// ---------- header: clock, mode, note ----------
function setNote(text, kind, html) {
  const n = $('note');
  if (html) n.innerHTML = text; else n.textContent = text || '';
  n.className = 'note' + (kind ? ' ' + kind : '');
  n.style.display = text ? '' : 'none';
}

function updateHeader(regions) {
  headerClock();
  const anyDay = regions.some(r => r.is_day);
  const anyGen = regions.some(r => r.generating);
  const sunny = '<button type="button" id="sunnyBtn" class="on">&#9654; Replay today\'s sun</button>';
  if (!regions.length) setNote('');
  else if (current === 'live' && !anyDay) setNote(`It is night in New Zealand right now, so solar panels make no power. ${sunny} plays the sunny hours of the past 24 hours.`, 'night', true);
  else if (current === 'live' && !anyGen) setNote(`Very little sun in New Zealand right now. ${sunny} plays the sunny hours of the past 24 hours.`, 'night', true);
  else if (isAhead() && !anyDay) setNote('It will be night at that hour, so the panels will make no power.', 'night');
  else if (!anyDay) setNote('Night at this hour, so the panels make no power. Move the slider or press Play.', 'night');
  else setNote('');
}
$('note').addEventListener('click', e => { if (e.target.id === 'sunnyBtn') watchSunnyDay(); });

// ---------- data loading ----------
async function getJson(url) {
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || ('HTTP ' + res.status));
  return body;
}

async function loadSnapshot(opts = {}) {
  const id = ++reqId;
  loadGrid();
  try {
    const data = await getJson('/api/snapshot?t=' + encodeURIComponent(current));
    if (id !== reqId) return;                       // a newer request already replaced this one
    drawRegions(data.regions);
    if (selected) markSelected(selected);
    updateHeader(data.regions);
    // On first open, show a region right away (the sunniest one now; Auckland at night) so the panel,
    // the power-flow animation and the chart are visible without a click.
    if (!selected && !selectedPoint && data.regions.length) {
      const best = data.regions.reduce((a, r) => ((r.predicted_kw_next || 0) > (a.predicted_kw_next || 0) ? r : a));
      const name = (best.predicted_kw_next || 0) > 0 ? best.name
        : (data.regions.find(r => r.name === 'Auckland') || data.regions[0]).name;
      selected = name;
      markSelected(name);
    }
  } catch (e) {
    if (id === reqId) setNote('Could not load data: ' + e.message, 'error');
  }
  if (opts.panel === false) return;
  if (selected) loadDetail(selected);
  else if (selectedPoint) loadPoint();
}

// Three modes: Replay the past 24 hours (what really happened), Live now, or Predict the next 1-3 hours.
let playTimer = null, lastReplay = null, DEFAULT_T = null, RECENT = true, AHEAD = [{ hours: 1 }], aheadH = 1;
const nzFmt = new Intl.DateTimeFormat('en-CA', { timeZone: 'Pacific/Auckland', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', hourCycle: 'h23' });
function nzKey(iso) { const p = {}; nzFmt.formatToParts(new Date(iso)).forEach(x => p[x.type] = x.value); return [`${p.year}-${p.month}-${p.day}`, +p.hour]; }
const NZ_SHORT = new Intl.DateTimeFormat('en-NZ', { timeZone: 'Pacific/Auckland', weekday: 'short', hour: '2-digit', minute: '2-digit', hour12: false });
const isAhead = () => String(current).startsWith('ahead:');
const isReplay = () => current !== 'live' && !isAhead();
function hoursAgo(t) {                     // the newest replay hour is the hour just before the live one
  if (!times.length) return null;
  return Math.round((Date.parse(times[times.length - 1]) + 3600e3 - Date.parse(t)) / 3600e3);
}
function hhmm(ms) { return new Intl.DateTimeFormat('en-NZ', { timeZone: 'Pacific/Auckland', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(ms)); }
// What the "expected ..." numbers refer to in the current mode.
function whenText() {
  if (!isAhead()) return 'next hour';
  const a = (lastAhead || []).find(x => x.hours === aheadH);
  return aheadH === 1 ? 'next hour' + (a ? ` (${a.from_local}-${a.to_local})` : '')
    : `in ${aheadH} hours` + (a ? ` (${a.from_local}-${a.to_local})` : '');
}
let lastAhead = null;
// Predict mode: the panel's headline number becomes the forecast for the chosen hour.
function withAhead(d) {
  if (d && d.ahead) lastAhead = d.ahead;
  if (!isAhead() || !d || !d.ahead) return d;
  const a = d.ahead.find(x => x.hours === aheadH);
  return a ? { ...d, predicted_kw_next: a.kw, generating: (a.kw || 0) > 1, charge_now: aheadH === 1 ? d.charge_now : false } : d;
}
function aheadButtons() {
  $('aheadBtns').innerHTML = AHEAD.map(a => `<button type="button" class="ah ${a.hours === aheadH ? 'on' : ''}" data-h="${a.hours}">+${a.hours} h</button>`).join(' ');
  const a = AHEAD.find(x => x.hours === aheadH) || {};
  const max = Math.max(...AHEAD.map(x => x.hours));
  $('aheadNote').innerHTML = (typeof a.pct_of_capacity === 'number'
    ? `Average error ${num(a.pct_of_capacity, 1)}% of your panels' size on unseen test months; ` +
      (a.better_than_weather_service_pct >= 10 ? `${num(a.better_than_weather_service_pct, 0)}% better than the weather service's own forecast.`
        : `about as good as the weather service's own forecast.`)
    : 'Most accurate one hour ahead.') +
    ` We stop at ${max} hour${max > 1 ? 's' : ''}: further ahead, the weather service is just as good.`;
}
$('aheadBtns').addEventListener('click', e => {
  const b = e.target.closest('.ah'); if (!b) return;
  aheadH = +b.dataset.h; setTime('ahead:' + aheadH);
});
function syncBar() {
  const mode = current === 'live' ? 'live' : isAhead() ? 'predict' : 'replay';
  $('liveBtn').classList.toggle('on', mode === 'live'); $('replayBtn').classList.toggle('on', mode === 'replay');
  $('predictBtn').classList.toggle('on', mode === 'predict');
  $('replayBox').hidden = mode !== 'replay'; $('predictBox').hidden = mode !== 'predict';
  $('modeBadge').className = 'badge ' + mode;
  $('modeBadge').innerHTML = { live: '<i></i>LIVE', replay: '&#9194; REPLAY', predict: '&#128302; PREDICT' }[mode];
  $('modeLabel').textContent = {
    live: 'The current hour, with the latest weather',
    replay: RECENT ? 'One of the past 24 hours: our forecast vs. what really happened' : 'A past hour from months the model never saw (live data unavailable)',
    predict: `Our forecast for ${aheadH === 1 ? 'the next hour' : 'the hour ' + aheadH + ' hours from now'}` }[mode];
  if (mode !== 'replay') stopPlay();
  if (mode === 'predict') aheadButtons();
  if (mode === 'replay') {
    const i = times.indexOf(current), ago = hoursAgo(current);
    $('slider').max = Math.max(0, times.length - 1); $('slider').value = Math.max(0, i);
    $('rHour').textContent = NZ_SHORT.format(new Date(current)) + (RECENT && ago !== null ? ` · ${ago} h ago` : '');
  }
  headerClock();
}
// Top-right clock, always in one format: the real NZ time in Live and Predict, the replayed hour (marked) in Replay.
function headerClock() {
  const rep = isReplay();
  $('nzTime').innerHTML = (rep ? '<small>Replaying</small> ' : '') + esc(nzString(rep ? current : new Date().toISOString())) +
    ` <small>${rep ? (RECENT ? 'New Zealand time (' + hoursAgo(current) + ' h ago)' : 'New Zealand time (a past day)') : 'New Zealand time now'}</small>`;
}
function setTime(t) {
  if (!t) return;
  current = t;
  if (isReplay()) lastReplay = t;
  syncBar();
  loadSnapshot();
}
function stopPlay() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
  $('rPlay').innerHTML = '&#9654; Play'; $('rPlay').classList.remove('on');
}
function step(dir) {
  const i = times.indexOf(current);
  if (i < 0 || !times[i + dir]) return false;
  setTime(times[i + dir]); return true;
}
// Play: hour by hour through the daylight of the past 24 hours (from the first hour after 05:00).
function firstMorning() {
  const i = times.findIndex(t => { const h = nzKey(t)[1]; return h >= 5 && h < 20; });
  return times[Math.max(0, i)];
}
function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  if (!isReplay()) setTime(lastReplay || DEFAULT_T);
  const h = nzKey(current)[1], i = times.indexOf(current);
  if (h < 5 || h >= 21 || i === times.length - 1) setTime(firstMorning());
  $('rPlay').innerHTML = '&#10074;&#10074; Pause'; $('rPlay').classList.add('on');
  playTimer = setInterval(() => {
    const i = times.indexOf(current);
    if (!times[i + 1] || nzKey(current)[1] >= 21) { stopPlay(); return; }
    setTime(times[i + 1]);
  }, 1600);
}
function watchSunnyDay() { stopPlay(); setTime(firstMorning()); togglePlay(); }
// The past 24 hours move with the clock: ask the server for the current list when Replay opens.
async function refreshTimes() {
  try {
    const info = await getJson('/api/times');
    if (info.times && info.times.length) { times = info.times; RECENT = info.replay_window !== 'history'; }
    if (info.default && times.includes(info.default)) DEFAULT_T = info.default;
  } catch (e) {}
}

$('liveBtn').onclick = () => setTime('live');
$('replayBtn').onclick = async () => {
  if (isReplay()) return;
  await refreshTimes();
  setTime(lastReplay && times.includes(lastReplay) ? lastReplay : DEFAULT_T);
};
$('predictBtn').onclick = () => { if (!isAhead()) setTime('ahead:' + aheadH); };
$('slider').addEventListener('input', () => {
  stopPlay();
  const t = times[+$('slider').value];
  if (!t) return;
  $('rHour').textContent = NZ_SHORT.format(new Date(t));
  clearTimeout(debounce);
  debounce = setTimeout(() => setTime(t), 150);
});
$('rPrev').onclick = () => { stopPlay(); step(-1); };
$('rNext').onclick = () => { stopPlay(); step(1); };
$('rPlay').onclick = togglePlay;

// ---------- detail panel ----------
function hourLabel(h) { return (typeof h === 'number') ? String(h).padStart(2, '0') + ':00' : String(h); }

function metricsHtml(m, label) {
  if (!m || typeof m !== 'object') return '';
  const rows = Object.entries(m).filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => `<tr><td>${esc(({ mae_kw: 'Average gap (kW)', rmse_kw: 'Gap that counts big misses more (kW)', mae_wm2: 'Average gap in sunlight (W/m²)' })[k] || k.replace(/_/g, ' '))}</td><td>${num(/_kw$/.test(k) ? S(v) : v, 2)}</td></tr>`).join('');
  return rows ? `<h3>${label || 'How accurate the forecast is here (test months)'}</h3><table class="m">${rows}</table>` +
    `<div class="sub">Gap between forecast and reality, for your ${SIZE} kW system. Smaller is better.</div>` : '';
}

async function loadDetail(name) {
  const id = ++detailId;
  const panel = $('panel');
  if (!panel.querySelector('canvas')) panel.innerHTML = `<div class="empty">Loading ${esc(name)}...</div>`;
  let d;
  try { d = await getJson('/api/detail?region=' + encodeURIComponent(name) + '&t=' + encodeURIComponent(current)); }
  catch (e) { if (id === detailId) panel.innerHTML = `<div class="empty">Could not load ${esc(name)}: ${esc(e.message)}</div>`; return; }
  if (id !== detailId) return;
  renderPanel(withAhead(d), esc(name), `Local day, hourly power of your ${SIZE} kW rooftop system`);
}

// ---------- any point in New Zealand ----------
let selectedPoint = null, pin = null;
async function loadPoint() {
  const p = selectedPoint, id = ++detailId, panel = $('panel');
  panel.innerHTML = `<div class="empty">Downloading the weather for ${num(p.lat, 2)}, ${num(p.lon, 2)} and running the model...</div>`;
  let d;
  try { d = await getJson(`/api/point?lat=${p.lat}&lon=${p.lon}&t=` + encodeURIComponent(current)); }
  catch (e) { if (id === detailId) panel.innerHTML = `<div class="empty">Could not load this point: ${esc(e.message)}</div>`; return; }
  if (id !== detailId) return;
  d = withAhead(d);
  const gen = d.generating ? `<b style="color:var(--good)">Your ${SIZE} kW panels: ${num(S(d.predicted_kw_next), kwd())} kW expected ${whenText()}</b>`
                           : `<b>Not generating: ${num(S(d.predicted_kw_next), kwd())} kW expected ${whenText()}</b>`;
  pin.setTooltipContent(`<b>${num(d.lat, 2)}, ${num(d.lon, 2)}</b><br>${gen}`);
  renderPanel(d, `Point ${num(d.lat, 2)}, ${num(d.lon, 2)}`,
    `${esc(d.local_time)} NZ &middot; ${gen}<br>Cloud ${num(d.cloud_now_pct, 0)}% &middot; ${num(d.temp_c)} &deg;C &middot; wind ${num(d.wind_kmh, 0)} km/h &middot; sunlight now ${num(d.ghi_now_wm2, 0)} W/m&sup2;<br>` +
    `Weather downloaded for this exact spot; the model uses the climate of the nearest region, <b>${esc(d.nearest_region)}</b> (${num(d.distance_km, 0)} km away).`);
}
function selectPoint(lat, lon) {
  selected = null; selectedPoint = { lat: +lat.toFixed(4), lon: +lon.toFixed(4) };
  markSelected(null);
  if (!pin) pin = L.circleMarker([lat, lon], { radius: 9, color: '#ffffff', weight: 3, fillColor: '#2563eb', fillOpacity: 1, bubblingMouseEvents: false })
    .addTo(map).bindTooltip('', { direction: 'top', offset: [0, -8], className: 'tip' });
  pin.setLatLng([lat, lon]);
  loadPoint();
  if (window.matchMedia('(max-width: 900px)').matches) $('panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
map.on('click', e => selectPoint(e.latlng.lat, e.latlng.lng));

let lastPanel = null;
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
// Numbers count up from 0 when a panel opens (skipped for reduced motion).
function countUp(el) {
  if (!el || REDUCED) return;
  const to = +el.dataset.v, dg = +el.dataset.d, t0 = performance.now();
  const step = now => { const k = Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - k, 3);
    el.textContent = num(to * e, dg); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

// ---------- charge your EV from your panels ----------
let EVCARS = {}, HOMEBATT = {}, HOME_CHARGER = 7.4, THREE_PHASE = 11;
let EVS = { car: 'tesla_model_y', start: 20, target: 80, charger: 7.4, batt: 'tesla_powerwall_3' };
try { Object.assign(EVS, JSON.parse(localStorage.getItem('solarcastEv') || '{}')); } catch (e) {}
const carIcon = c => `<svg viewBox="0 0 38 22"><path d="M2 12 L8 3 L26 3 L33 12 L38 13 L38 20 L0 20 L0 13 Z" fill="${c}"/><circle cx="9" cy="20" r="3.5" fill="#1c2733"/><circle cx="29" cy="20" r="3.5" fill="#1c2733"/></svg>`;
function evCardHtml() {
  const cars = Object.entries(EVCARS);
  if (!cars.length) return '';
  return `<h3>Charge your EV from your ${SIZE} kW panels</h3>
  <div class="evc">
    <div class="evcars">${carIcon('#1f8f4d')}<input type="search" id="evFind" placeholder="Find your car: type a brand or model (Tesla, BYD, Leaf...)" aria-label="Search cars" autocomplete="off">
      <select id="evCar" aria-label="Your car">${carOptions('')}</select><span id="evCount" class="sub"></span></div>
    <div class="evin">
      <span>Battery now</span><input type="range" id="evStart" min="0" max="95" step="5" value="${EVS.start}"><b id="evStartV">${EVS.start}%</b>
      <span>Charge to</span><input type="range" id="evTarget" min="10" max="100" step="5" value="${EVS.target}"><b id="evTargetV">${EVS.target}%</b>
      <span>Charger</span><select id="evCharger">
        <option value="${HOME_CHARGER}" ${EVS.charger == HOME_CHARGER ? 'selected' : ''}>${HOME_CHARGER} kW home wall charger (single-phase, most NZ homes)</option>
        <option value="${THREE_PHASE}" ${EVS.charger == THREE_PHASE ? 'selected' : ''}>${THREE_PHASE} kW wall charger (three-phase)</option>
        <option value="2.3" ${EVS.charger == 2.3 ? 'selected' : ''}>2.3 kW normal wall socket (10 A)</option></select>
    </div>
    <div id="evOut"><div class="sub">Working it out...</div></div>
    ${Object.keys(HOMEBATT).length ? `<div class="hb"><h4>&#128267; Store the left-over sun in a home battery</h4>
      <select id="evBatt" aria-label="Home battery"><option value="">No home battery</option>${Object.entries(HOMEBATT).map(([k, b]) =>
        `<option value="${k}" ${k === EVS.batt ? 'selected' : ''}>${esc(b.name)} &middot; ${b.capacity_kwh} kWh</option>`).join('')}</select>
      <div id="hbOut"></div></div>` : ''}
  </div>`;
}
// Cars grouped by brand; `q` filters by brand or model ("tesla", "byd seal", "leaf").
function carOptions(q) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean), groups = {};
  Object.entries(EVCARS).forEach(([k, c]) => {
    const hay = (c.name + ' ' + (c.brand || '') + ' ' + c.note).toLowerCase();
    if (words.every(w => hay.includes(w))) (groups[c.brand || 'Other'] = groups[c.brand || 'Other'] || []).push([k, c]);
  });
  return Object.entries(groups).map(([b, list]) => `<optgroup label="${esc(b)}">` + list.map(([k, c]) =>
    `<option value="${k}" ${k === EVS.car ? 'selected' : ''}>${esc(c.name)} &middot; ${c.battery_kwh} kWh${c.note ? ' \u2014 ' + esc(c.note) : ''}</option>`).join('') + '</optgroup>').join('');
}
let evTimer = null, evReq = 0;
function evBind(d) {
  if (!$('evOut')) return;
  const save = () => { try { localStorage.setItem('solarcastEv', JSON.stringify(EVS)); } catch (e) {} };
  $('evCar').onchange = () => { EVS.car = $('evCar').value; save(); evRun(d); };
  $('evFind').oninput = () => {
    const q = $('evFind').value, sel = $('evCar');
    sel.innerHTML = carOptions(q);
    const n = sel.options.length;
    $('evCount').textContent = q ? (n ? n + (n === 1 ? ' car' : ' cars') : 'No match. Try a brand: Tesla, BYD, MG, Kia, Hyundai...') : '';
    if (n && sel.value !== EVS.car) { EVS.car = sel.value; save(); evRun(d); }
  };
  if ($('evBatt')) $('evBatt').onchange = () => { EVS.batt = $('evBatt').value; save(); evRun(d); };
  const slide = () => {
    let a = +$('evStart').value, b = +$('evTarget').value;
    if (b <= a) { b = Math.min(100, a + 5); $('evTarget').value = b; }
    EVS.start = a; EVS.target = b; $('evStartV').textContent = a + '%'; $('evTargetV').textContent = b + '%'; save();
    clearTimeout(evTimer); evTimer = setTimeout(() => evRun(d), 200);
  };
  $('evStart').oninput = slide; $('evTarget').oninput = slide;
  $('evCharger').onchange = () => { EVS.charger = +$('evCharger').value; save(); evRun(d); };
  evRun(d);
}
async function evRun(d) {
  const id = ++evReq, out = $('evOut');
  let nh = d.next_local_hour !== undefined ? d.next_local_hour : (d.hours || [])[0] || 0, whole = false;
  const ask = async h => {
    const res = await fetch('/api/ev_plan', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hours: d.hours || [], solar_kw: (d.predicted_kw || []).map(v => v === null ? null : S(v)),
        next_hour: h, car: EVS.car, start_pct: EVS.start, target_pct: EVS.target, charger_kw: EVS.charger,
        home_battery: HOMEBATT[EVS.batt] ? EVS.batt : null }) });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || 'HTTP ' + res.status);
    return body;
  };
  let p;
  try {
    p = await ask(nh);
    // No sun left today (evening, or a judge abroad seeing NZ at night): show the whole day as if plugged in at sunrise.
    if (!p.done && !p.hours.some(r => r.car_kw > 0) && nh > 0) {
      const all = await ask(0);
      if (all.hours.some(r => r.car_kw > 0)) { p = all; nh = 0; whole = true; }
    }
  } catch (e) { if (id === evReq && out) out.innerHTML = `<div class="sub">Could not estimate: ${esc(e.message)}</div>`; return; }
  if (id !== evReq || !$('evOut')) return;
  const sunLeft = p.hours.some(r => r.car_kw > 0), name = esc(p.car_name);
  const when = whole ? `<div class="sub" style="margin-bottom:2px">No sun left today, so this shows the whole day${d.local_date ? ' (' + esc(d.local_date) + ')' : ''}, plugged in from sunrise.</div>` : '';
  const head = when + (p.done
    ? `<div class="evhead ok">&#9728; ${p.start_pct}% &rarr; ${p.target_pct}% by ${p.finish_local_time} from your panels alone</div>`
    : sunLeft
      ? `<div class="evhead part">By the end of the day your panels get it to ${p.end_pct}% (+${p.km_added} km). Still needed for ${p.target_pct}%: ${num(p.still_needed_kwh)} kWh (about NZD ${num(p.still_needed_grid_cost_nzd, 2)} from the grid) or tomorrow's sun.</div>`
      : `<div class="evhead part">Not enough sun ${whole ? 'this day' : 'left today'} to charge the ${name} from ${SIZE} kW of panels (an EV charger needs at least 1.4 kW to spare).</div>`);
  const rows = p.hours.filter(r => r.solar_kw > 0.05);
  let bars = '';
  if (rows.length) {
    const W = 330, H = 70, top = Math.max(...rows.map(r => r.solar_kw), 0.1), bw = W / rows.length;
    bars = `<svg class="evbars" viewBox="0 0 ${W} ${H + 14}" role="img" aria-label="Where each hour's solar power goes">` +
      rows.map((r, i) => { const x = i * bw + 2, w = Math.max(2, bw - 4), sc = H / top; let y = H, out = '';
        [[r.home_kw, '#94a3b8'], [r.car_kw, '#1f8f4d'], [r.unused_kw, '#fcd38d']].forEach(([v, c]) => {
          const h = Math.max(0, v) * sc; if (h > 0.2) { y -= h; out += `<rect class="bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="${c}" style="animation-delay:${i * 45}ms"/>`; } });
        return out + `<text x="${(x + w / 2).toFixed(1)}" y="${H + 11}" text-anchor="middle">${String(r.hour).padStart(2, '0')}</text>`; }).join('') + '</svg>' +
      `<div class="evkey"><span><i style="background:#94a3b8"></i>home (${num(0.5, 1)} kW)</span><span><i style="background:#1f8f4d"></i>into the car</span><span><i style="background:#fcd38d"></i>left over</span></div>`;
  }
  out.innerHTML = `
    <div class="batt"><div class="b0" style="width:${p.start_pct}%"></div><div class="b1" style="left:${p.start_pct}%"></div>
      <div class="tgt" style="left:${p.target_pct}%"></div><div class="lbl">${p.start_pct}% &rarr; <span class="cu" data-v="${p.end_pct}" data-d="0">${p.end_pct}</span>%</div></div>
    ${head}
    <div class="stats" style="margin-top:6px">
      <div class="stat"><b>${p.done ? num(p.charging_hours) + ' h' : '+' + p.km_added + ' km'}</b><span>${p.done ? `of solar charging (+${p.km_target} km range)` : 'range added from solar today'}</span></div>
      <div class="stat"><b>NZD ${num(p.saved_nzd, 2)}</b><span>grid power you don't buy</span></div>
      <div class="stat"><b>${num(p.grid_only_hours)} h</b><span>same charge plugged into the grid at ${num(p.charger_kw)} kW (NZD ${num(p.grid_only_cost_nzd, 2)})</span></div>
    </div>
    ${bars}
    <div class="sub" style="margin-top:4px">${name}: ${p.battery_kwh} kWh battery, charges at up to ${num(p.charger_kw)} kW here. Uses this day's forecast from ${whole ? 'sunrise' : String(nh).padStart(2, '0') + ':00'}, a 0.5 kW home load first, 10% charging loss. An estimate: real cars and weather vary.</div>`;
  requestAnimationFrame(() => { const b = out.querySelector('.b1'); if (b) b.style.width = Math.max(0, p.end_pct - p.start_pct) + '%'; });
  countUp(out.querySelector('.batt .cu'));
  hbShow(p);
}
function hbShow(p) {
  const o = $('hbOut'), h = p.home_battery;
  if (!o) return;
  if (!h) { o.innerHTML = '<div class="sub">Pick a battery to see how much of today\'s left-over solar it could keep for the evening.</div>'; return; }
  if (!h.stored_kwh) { o.innerHTML = `<div class="sub">No solar left over ${p.done || p.hours.some(r => r.car_kw > 0) ? 'after the car' : 'today'} to store in the ${esc(h.name)}.</div>`; return; }
  const pct = Math.round(h.stored_kwh / h.capacity_kwh * 100);
  o.innerHTML = `<div class="batt hbb"><div class="b1" style="left:0"></div><div class="lbl">${num(h.stored_kwh)} of ${h.capacity_kwh} kWh</div></div>
    <div class="stats" style="margin-top:6px">
      <div class="stat"><b>${h.full_by ? 'Full by ' + h.full_by : pct + '% full'}</b><span>stored from left-over sun</span></div>
      ${h.to_car_kwh > 0 ? `<div class="stat"><b>${p.end_pct}% &rarr; ${h.car_end_pct}%</b><span>the car tonight, from the battery (${num(h.to_car_kwh)} kWh)</span></div>` : ''}
      <div class="stat"><b>${num(h.home_hours)} h</b><span>of evening home power (about ${num(1, 0)} kW), no grid</span></div>
      <div class="stat"><b>NZD ${num(h.saved_nzd, 2)}</b><span>more grid power you don't buy</span></div>
    </div>
    <div class="sub">The ${esc(h.name)} starts the morning empty, takes up to its charging power from what the home and car don't use, and gives back about 90% in the evening: first to the car if it still needs charge, then to the home. Typical published sizes; an estimate.</div>`;
  requestAnimationFrame(() => { const b = o.querySelector('.b1'); if (b) b.style.width = Math.min(100, pct) + '%'; });
}
// Animated picture of where the next hour's power goes: sun -> your panels -> home (-> EV when the example says charge).
// The dots move faster and the sun is brighter when the panels give more of their rated power; at night the home runs on the grid.
function flowSvg(d) {
  const kw = typeof d.predicted_kw_next === 'number' ? d.predicted_kw_next : 0;
  const share = Math.max(0, Math.min(1, kw / REF));
  const on = share >= 0.02, car = on && !!d.charge_now;
  const speed = (2.6 - 2.1 * share).toFixed(2) + 's';
  const rays = [0, 45, 90, 135, 180, 225, 270, 315].map(a => {
    const r = a * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
    return `<line x1="${(30 + 15 * c).toFixed(1)}" y1="${(26 + 15 * s).toFixed(1)}" x2="${(30 + 21 * c).toFixed(1)}" y2="${(26 + 21 * s).toFixed(1)}"/>`;
  }).join('');
  const sun = on
    ? `<g opacity="${(0.35 + 0.65 * share).toFixed(2)}"><g class="sunrays" stroke="#f5a623" stroke-width="2.5" stroke-linecap="round">${rays}</g>
       <circle cx="30" cy="26" r="11" fill="#f5b400"/></g>`
    : `<path d="M36 14 a13 13 0 1 0 6 22 a10 10 0 1 1 -6 -22z" fill="#8a9bb8"/>`;
  const juice = (p, cls) => `<path class="wire" d="${p}"/><path class="${cls}" d="${p}" style="--speed:${speed}"/>`;
  return `<svg class="flow" viewBox="0 0 330 118" role="img" aria-label="${on
      ? `Power flow: your ${SIZE} kW panels send about ${num(S(kw), kwd())} kW to your home${car ? ' and the EV' : ''} next hour`
      : 'No solar power next hour: the home runs on the grid'}">
    ${sun}
    ${on ? juice('M52 40 L92 62', 'juice') : '<path class="wire" d="M52 40 L92 62"/>'}
    <g transform="translate(92 54)"><path d="M0 22 L12 0 L44 0 L32 22 Z" fill="#1e3a5f"/>
      <path d="M11 2 L5 20 M22 0 L16 22 M33 0 L27 22 M6 11 L38 11" stroke="#7fa7d6" stroke-width="1"/>
      <line x1="22" y1="22" x2="22" y2="34" stroke="#6b7684" stroke-width="2"/></g>
    <text x="114" y="104" text-anchor="middle">Your ${SIZE} kW</text>
    ${on ? juice('M134 70 C170 70 180 70 214 70', 'juice') : '<path class="wire" d="M134 70 C170 70 180 70 214 70"/>'}
    ${on ? `<text class="kw" x="174" y="60" text-anchor="middle">${num(S(kw), kwd())} kW</text>` : ''}
    <g transform="translate(214 46)"><path d="M0 18 L22 0 L44 18 Z" fill="#c2410c"/><rect x="5" y="18" width="34" height="26" fill="#f1e4d3"/>
      <rect x="17" y="28" width="10" height="16" fill="#8b5a2b"/><rect x="8" y="22" width="7" height="7" fill="${on ? '#fde68a' : '#93c5fd'}"/></g>
    <text x="236" y="104" text-anchor="middle">Home</text>
    ${car ? juice('M258 82 C276 86 280 92 292 92', 'juice') : '<path class="wire" d="M258 82 C276 86 280 92 292 92"/>'}
    <g transform="translate(288 80)" opacity="${car ? 1 : 0.45}"><path d="M2 12 L8 3 L26 3 L33 12 L38 13 L38 20 L0 20 L0 13 Z" fill="${car ? '#1f8f4d' : '#6b7684'}"/>
      <circle cx="9" cy="21" r="4" fill="#1c2733"/><circle cx="29" cy="21" r="4" fill="#1c2733"/></g>
    <text x="328" y="114" text-anchor="end">${car ? 'EV charging' : 'EV'}</text>
    ${on ? '' : `<g transform="translate(176 6)" stroke="#243b6b" stroke-width="2" fill="none"><path d="M10 0 L2 34 M10 0 L18 34 M4 14 L16 14 M3 24 L17 24"/></g>
      <text x="200" y="20">Grid</text>${juice('M186 42 C200 56 210 60 222 62', 'grid-juice')}`}
  </svg>`;
}
function yoursHtml(d) {
  const sum = a => (a || []).reduce((t, v) => t + (typeof v === 'number' ? v : 0), 0);
  const day = S(sum(d.predicted_kw)), done = (d.actual_kw || []).some(v => typeof v === 'number') ? S(sum(d.actual_kw)) : null;
  return `<div class="yours"><div class="sub">Your ${SIZE} kW panels, ${whenText()}</div>
    ${flowSvg(d)}
    <div class="big"><span class="cu" data-v="${S(d.predicted_kw_next) || 0}" data-d="${kwd()}">${num(S(d.predicted_kw_next), kwd())}</span> kW <small>&asymp; ${num(S(d.predicted_kw_next), kwd())} kWh over the hour</small></div>
    <div class="stats" style="margin-top:8px">
      <div class="stat"><b>${num(day)}</b><span>kWh forecast for this day</span></div>
      <div class="stat"><b>NZD ${num(day * PRICE, 2)}</b><span>worth at NZD ${num(PRICE, 2)}/kWh</span></div>
      <div class="stat"><b>${done === null ? '--' : num(done)}</b><span>kWh actual (from weather data)</span></div>
    </div></div>`;
}
function renderPanel(d, titleHtml, subHtml) {
  lastPanel = [d, titleHtml, subHtml];
  const panel = $('panel');
  const yes = !!d.charge_now;
  const mlabel = d.is_point ? `How accurate the forecast is in the nearest region (${esc(d.nearest_region)})` : null;
  panel.innerHTML = `
    <h2>${titleHtml}</h2>
    <div class="sub">${subHtml}</div>
    ${yoursHtml(d)}
    <div class="chartbox"><canvas id="chart"></canvas></div>
    ${evCardHtml()}
    <h3>Example use: should a workshop charge its EV now?</h3>
    <div class="sub" style="margin-bottom:6px">One way to use the forecast. This example is fixed: a workshop with ${REF} kW of panels, a 6 kW load and an 11 kW EV charger.</div>
    <div class="advice ${yes ? 'yes' : 'no'}"><strong>${yes ? 'Charge now' : 'Wait'}</strong><br>${esc(d.reason || '')}</div>
    <h3>Savings today vs charging on arrival</h3>
    <div class="stats">
      <div class="stat"><b>${num(d.grid_kwh_saved)}</b><span>kWh from grid avoided</span></div>
      <div class="stat"><b>NZD ${num(d.money_saved_nzd, 2)}</b><span>money saved</span></div>
      <div class="stat"><b>${num(d.co2_saved_kg)}</b><span>kg CO2 avoided</span></div>
    </div>
    ${d.grid_kwh_saved_vs_fixed !== undefined ? `<div class="sub" style="margin-top:6px">Against a tougher habit (always charge 10:00-14:00): ${num(d.grid_kwh_saved_vs_fixed)} kWh, NZD ${num(d.money_saved_vs_fixed_nzd, 2)} saved today. Over the whole test period the forecast saves 1.5 kWh/day vs that habit and 11.1 kWh/day vs charging on arrival.</div>` : ''}
    ${d.assumptions ? `<div class="sub" style="margin-top:6px">Assumptions: ${esc(d.assumptions)}</div>` : ''}
    ${metricsHtml(d.metrics, mlabel)}`;
  drawChart(d);
  countUp(panel.querySelector('.yours .big .cu'));
  evBind(d);
}

function drawChart(d) {
  if (chart) { chart.destroy(); chart = null; }
  const labels = (d.hours || []).map(hourLabel);
  chart = new Chart($('chart'), {
    data: {
      labels,
      datasets: [
        { type: 'line', label: 'What the panels made (from weather data)', data: (d.actual_kw || []).map(S), borderColor: '#1f8f4d', backgroundColor: '#1f8f4d', borderWidth: 2.5, pointRadius: 0, tension: 0.25, spanGaps: false, order: 1 },
        { type: 'line', label: 'Our forecast', data: (d.predicted_kw || []).map(S), borderColor: '#e8890c', backgroundColor: '#e8890c', borderWidth: 2.5, borderDash: [6, 4], pointRadius: 0, tension: 0.25, order: 2 },
        { type: 'line', label: 'Simple guess: same as last hour', data: (d.persistence_kw || []).map(S), borderColor: '#6b7684', backgroundColor: '#6b7684', borderWidth: 1.5, borderDash: [2, 3], pointRadius: 0, tension: 0.25, order: 3 },
        { type: 'bar', label: 'Cloud cover (%)', data: d.cloud_cover_pct || [], yAxisID: 'y2', backgroundColor: 'rgba(93,107,122,.15)', order: 4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 14, font: { size: 11 } } } },
      scales: {
        y: { title: { display: true, text: `kW (your ${SIZE} kW system)` }, beginAtZero: true, suggestedMax: SIZE / 5 },
        y2: { position: 'right', min: 0, max: 100, display: false },
        x: { ticks: { maxTicksLimit: 8, font: { size: 11 } } },
      },
    },
  });
}

function markSelected(name) {
  Object.entries(markers).forEach(([n, m]) => {
    m.setStyle({ radius: n === name ? 17 : 14, color: n === name ? '#1c2733' : '#ffffff', weight: n === name ? 3 : 2 });
    const el = m.getElement && m.getElement(); if (el) el.classList.toggle('selpulse', n === name);
  });
}
function selectRegion(name) {
  selected = name; selectedPoint = null;
  if (pin) { map.removeLayer(pin); pin = null; }
  markSelected(name);
  loadDetail(name);
  if (window.matchMedia('(max-width: 900px)').matches) $('panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- tabs and Model view ----------
const mcharts = {};
function showTab(which) {
  const model = which === 'model';
  $('mapView').style.display = model ? 'none' : '';
  $('modelView').style.display = model ? 'block' : 'none';
  $('tabMap').classList.toggle('on', !model);
  $('tabModel').classList.toggle('on', model);
  if (!model) setTimeout(() => map.invalidateSize(), 50);
  else renderModel();
}
$('tabMap').onclick = () => showTab('map');
$('tabModel').onclick = () => showTab('model');

const MODEL_COLORS = ['#e8890c', '#6b7684', '#4a90d9', '#1f8f4d', '#8e44ad'];
// Plain-language names for the comparison methods and the model's inputs.
const PLAIN_MODEL = { 'SolarCast model': 'SolarCast (our forecast)', 'Persistence': 'Simple guess: same power as last hour',
  'Clear-sky persistence': 'Smarter guess: same cloudiness as last hour' };
const pm = m => PLAIN_MODEL[m] || m;
const PLAIN_FEATURE = {
  cs_target: 'Sunlight on a clear day, next hour', elev_target: 'How high the sun will be', azim_target: 'Where the sun will be in the sky',
  kt_t: 'How clear the sky is now', kt_roll24: 'How clear the sky was over the last day', cs_t: 'Sunlight on a clear day, now',
  cloud_t: 'Cloud cover now', d_cloud_1h: 'Clouds coming or going (last hour)', ghi_t: 'Sunlight now',
  ghi_lag1: 'Sunlight 1 hour ago', ghi_lag2: 'Sunlight 2 hours ago', ghi_lag23: 'Sunlight this time yesterday',
  ghi_roll3: 'Average sunlight, last 3 hours', ghi_roll24: 'Average sunlight, last day', d_ghi_1h: 'Sunlight change in the last hour',
  dni_t: 'Direct sunbeams now', dhi_t: 'Light scattered by clouds now', temp_t: 'Temperature now', d_temp_1h: 'Temperature change',
  humidity_t: 'Humidity now', d_humidity_1h: 'Humidity change', wind_t: 'Wind now', precip_t: 'Rain now',
  hour_sin: 'Time of day', hour_cos: 'Time of day', doy_sin: 'Time of year', doy_cos: 'Time of year',
  region_id: 'Which region', lat: 'How far south', fc_ghi_target: "Weather service's sunlight forecast",
  fc_cloud_target: "Weather service's cloud forecast", fc_temp_target: "Weather service's temperature forecast",
  fc_dni_target: "Weather service's sunbeam forecast", fc_kt_target: "Weather service's clear-sky forecast",
  clear_sky_ghi_target: 'Sunlight on a clear day, next hour', sun_elevation_target: 'How high the sun will be', cloud_cover_t: 'Cloud cover now' };
const PLAIN_PERIOD = { train: 'Learned from', val: 'Fine-tuned on', test: 'Tested on (months it never saw)' };
function mchart(id, config) {
  if (mcharts[id]) mcharts[id].destroy();
  mcharts[id] = new Chart($(id), config);
}
function renderModel() {
  if (!report) { $('mPeriods').textContent = 'Model report not available.'; return; }
  const p = report.periods || {};
  $('mPeriods').innerHTML = Object.entries(p).map(([k, v]) =>
    `<b>${esc(PLAIN_PERIOD[k] || k)}</b>: ${esc(Array.isArray(v) ? v.join(' to ') : v)}`).join(' &nbsp;|&nbsp; ') +
    `<br>In short: on average the forecast is off by about 2% of what the panels can make at full sun. kW numbers on this tab are for a ${REF} kW system; for your ${SIZE} kW panels divide them by ${num(REF / SIZE, 1)}.`;

  const prev = report.previous;
  const mv = (o) => o ? `${num(o.mae_kw, 2)} kW (${num(o.mae_wm2, 1)} W/m&sup2;)` : '-';
  const cur = (report.overall || []).find(o => o.model === 'SolarCast model');
  $('hVersion').textContent = 'Model version: ' + (report.version || 'unknown');
  $('mVersion').innerHTML = (report.changes ? `<p>${esc(report.changes)}</p>` : '') +
    (prev && cur ? `<p><b>Test error, daytime:</b> ${esc(report.version || 'current')} ${mv(cur)} vs ${esc(prev.version)} ${mv(prev.test)}.</p>` : '') +
    (report.experiments ? '<p><b>How we got here (validation period, bar: ' + num(report.experiments.bar_wm2, 0) + ' W/m&sup2; better than ' +
      esc((report.previous || {}).version || 'v1') + '):</b></p><ul>' +
      ['v2', 'v3', 'v4'].filter(k => report.experiments[k]).map(k => `<li><b>${k}</b>: ${esc(report.experiments[k].summary)}</li>`).join('') + '</ul>' : '');

  const overall = report.overall || [];
  const names = overall.map(o => pm(o.model));
  const colors = names.map((_, i) => MODEL_COLORS[i % MODEL_COLORS.length]);
  const bar = (id, key, title) => mchart(id, { type: 'bar',
    data: { labels: names, datasets: [{ label: title, data: overall.map(o => o[key]), backgroundColor: colors }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } } });
  bar('cOverallKw', 'mae_kw', 'MAE (kW)');
  bar('cOverallWm', 'mae_wm2', 'MAE (W/m2)');
  const cols = [['mae_kw', 'Average gap (kW)'], ['mae_wm2', 'Average gap (W/m&sup2;)'], ['rmse_kw', 'Gap that counts big misses more (kW)']];
  $('mTable').innerHTML = '<table class="m"><tr><th>Method</th>' + cols.map(c => `<th>${c[1]}</th>`).join('') + '</tr>' +
    overall.map(o => `<tr><td>${esc(pm(o.model))}</td>` + cols.map(c => `<td>${num(o[c[0]], 2)}</td>`).join('') + '</tr>').join('') + '</table>';

  const rows = report.per_region || [];
  const regions = [...new Set(rows.map(r => r.region))];
  const models = [...new Set(rows.map(r => r.model))];
  mchart('cRegion', { type: 'bar',
    data: { labels: regions, datasets: models.map((m, i) => ({ label: pm(m), backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length],
      data: regions.map(rg => { const r = rows.find(x => x.region === rg && x.model === m); return r ? r.mae_kw : null; }) })) },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'Average gap (kW)' } }, x: { ticks: { maxRotation: 60, minRotation: 40 } } } } });

  const s = report.sample || {};
  $('hSample').textContent = 'Sample week' + (s.region ? ': ' + s.region : '');
  mchart('cSample', { type: 'line',
    data: { labels: (s.times || []).map(nzString), datasets: [
      { label: 'What the panels made', data: s.actual_kw || [], borderColor: '#1f8f4d', borderWidth: 2, pointRadius: 0, tension: 0.2 },
      { label: 'Our forecast', data: s.predicted_kw || [], borderColor: '#e8890c', borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: 0.2 },
      { label: 'Simple guess: same as last hour', data: s.persistence_kw || [], borderColor: '#6b7684', borderWidth: 1.2, borderDash: [2, 3], pointRadius: 0, tension: 0.2 } ] },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom' } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'kW' } }, x: { ticks: { maxTicksLimit: 8 } } } } });

  // Merge inputs that mean the same thing for a reader (e.g. the two "time of day" columns), show each as a % share.
  const agg = {};
  (report.importance || []).forEach(i => { const k = PLAIN_FEATURE[i.feature] || i.feature.replace(/_/g, ' '); agg[k] = (agg[k] || 0) + Math.max(0, i.importance); });
  const tot = Object.values(agg).reduce((a, b) => a + b, 0) || 1;
  const imp = Object.entries(agg).map(([k, v]) => [k, v / tot * 100]).sort((a, b) => b[1] - a[1]).slice(0, 10);
  mchart('cImp', { type: 'bar',
    data: { labels: imp.map(i => i[0]), datasets: [{ label: 'Share', data: imp.map(i => +i[1].toFixed(1)), backgroundColor: '#e8890c', borderRadius: 4 }] },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => ` ${c.parsed.x}% of the decisions` } } },
      scales: { x: { beginAtZero: true, grace: '8%', ticks: { callback: v => v + '%' } } } } });
}

async function loadModel() {
  try { report = await getJson('/api/model'); } catch (e) { report = null; }
  prepareAccuracy(); drawLegend();
  if ($('modelView').style.display === 'block') renderModel();
}

// ---------- SolarCast AI chat ----------
const chatHistory = [];                 // [{role, content}], kept in this browser tab only
let chatState = null, chatBusy = false, chatTimer = null;
function fmtAnswer(s) { return esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>'); }
function addMsg(role, html, cls = '') {
  const d = document.createElement('div');
  d.className = `msg ${role} ${cls}`.trim();
  d.innerHTML = html;
  $('chatLog').appendChild(d);
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
  return d;
}
function chatContext() {
  const when = !isReplay() ? 'live data for the current hour'
    : `live data; ask about "the replayed hour" to get the one on the map (${nzString(current)} NZ)`;
  $('chatCtx').textContent = `Answers use your ${SIZE} kW panels and ${when}` + (selected ? ` and the selected region (${selected}).`
    : selectedPoint ? ` and the clicked point (${selectedPoint.lat.toFixed(2)}, ${selectedPoint.lon.toFixed(2)}).` : '.');
}
async function pollChat() {
  clearTimeout(chatTimer);
  try { chatState = await getJson('/api/chat/status'); } catch (e) { chatState = { state: 'off' }; }
  const s = chatState.state;
  if (s === 'off') { $('chatBtn').style.display = 'none'; $('chat').classList.remove('open'); return; }
  $('chatBtn').style.display = $('chat').classList.contains('open') ? 'none' : 'block';
  const el = $('chatStatus');
  el.className = s === 'ready' ? 'ready' : '';
  el.textContent = s === 'ready' ? `AI model: ${chatState.model || 'local model'} (${chatState.device || 'local'}), offline, answers only from this data.`
    : s === 'templates' ? 'AI language model not available: answers come from fixed templates built from the data.'
    : (chatState.detail || 'Starting the AI model...') + ' Until it is ready, answers use fixed templates.';
  if (!['ready', 'templates'].includes(s)) chatTimer = setTimeout(pollChat, 4000);
}
function openChat(open) {
  $('chat').classList.toggle('open', open);
  $('chatBtn').style.display = open ? 'none' : 'block';
  if (open) { chatContext(); pollChat(); $('chatInput').focus(); }
}
async function sendChat(q) {
  q = (q || '').trim();
  if (!q || chatBusy) return;
  chatBusy = true; $('chatSend').disabled = true; $('chatInput').value = '';
  addMsg('user', esc(q));
  const wait = addMsg('bot', 'Thinking... (a few seconds on a GPU, longer on a CPU)', 'wait');
  chatContext();
  try {
    const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, t: current, region: selected, point: selectedPoint, system_kw: SIZE, history: chatHistory.slice(-6) }) });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || ('HTTP ' + res.status));
    const src = body.data_time === 'live' ? 'live data' : body.data_time ? `the replayed hour ${nzString(body.data_time)}` : '';
    const tag = body.mode === 'llm' ? `AI answer from ${src || 'the dashboard data'}` : body.mode === 'templates' ? `template answer from ${src || 'the data'}`
      : body.mode === 'refused' ? 'outside what this assistant answers' : '';
    wait.className = 'msg bot';
    wait.innerHTML = fmtAnswer(body.answer || '') + (tag ? `<span class="tag">${esc(tag)}</span>` : '');
    chatHistory.push({ role: 'user', content: q }, { role: 'assistant', content: body.answer || '' });
  } catch (e) {
    wait.className = 'msg bot'; wait.textContent = 'Sorry, something went wrong: ' + e.message;
  }
  chatBusy = false; $('chatSend').disabled = false; $('chatInput').focus();
}
$('chatBtn').onclick = () => openChat(true);
$('chatClose').onclick = () => openChat(false);
$('chatClear').onclick = () => { chatHistory.length = 0; [...$('chatLog').children].slice(1).forEach(n => n.remove()); };
$('chatForm').onsubmit = e => { e.preventDefault(); sendChat($('chatInput').value); };
[...$('chatChips').children].forEach(b => b.onclick = () => sendChat(b.textContent));
pollChat();

// ---------- panel size ----------
// A preset from the list, or any size typed in kW (0.5 - 1000). Remembered in this browser.
function showSize() {
  $('sizeNum').value = String(SIZE);
  $('sizeSel').value = [...$('sizeSel').options].some(o => o.value === String(SIZE)) ? String(SIZE) : 'custom';
}
function setSize(v) {
  v = Math.round(Math.min(1000, Math.max(0.5, +v || 5)) * 10) / 10;
  if (v === SIZE) { showSize(); return; }
  SIZE = v; showSize();
  try { localStorage.setItem('solarcastSize', String(SIZE)); } catch (e) {}
  drawLegend(); drawRegions(lastRegions); drawGrid(lastGrid); if (lastGrid) renderBrief(lastGrid);
  if (selectedPoint && !selected) loadPoint();
  else if (lastPanel && selected) renderPanel(lastPanel[0], lastPanel[1], `Local day, hourly power of your ${SIZE} kW rooftop system`);
  if ($('modelView').style.display === 'block') renderModel();
  chatContext();
}
if (!(SIZE >= 0.5 && SIZE <= 1000)) SIZE = 5;
showSize();
$('sizeSel').onchange = () => { if ($('sizeSel').value === 'custom') $('sizeNum').focus(); else setSize($('sizeSel').value); };
$('sizeNum').onchange = () => setSize($('sizeNum').value);

// ---------- start ----------
drawLegend();
loadModel();
(async function init() {
  try {
    const info = await getJson('/api/times');
    times = info.times || [];
    if (info.reference_kw) REF = info.reference_kw;
    if (info.grid_price_nzd) PRICE = info.grid_price_nzd;
    if (info.home_batteries) HOMEBATT = info.home_batteries;
    if (info.ev_cars) { EVCARS = info.ev_cars; if (!EVCARS[EVS.car]) EVS.car = Object.keys(EVCARS)[0]; HOME_CHARGER = info.home_charger_kw || HOME_CHARGER; THREE_PHASE = info.three_phase_charger_kw || THREE_PHASE; }
    RECENT = info.replay_window !== 'history';
    if (info.ahead && info.ahead.length) AHEAD = info.ahead;
    DEFAULT_T = times.includes(info.default) ? info.default : times[0];   // Replay starts at the sunniest past hour
    setTime('live');                                 // but the page always opens on the current hour (Live)
  } catch (e) { setNote('Could not load the time list: ' + e.message, 'error'); }
})();
</script>
</body>
</html>
"""
