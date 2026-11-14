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

from solarcast.nz import config

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
        return _json({"times": list(service.available_times()), "default": service.default_time(),
                      "reference_kw": config.SYSTEM_KW, "grid_price_nzd": config.GRID_PRICE_NZD})

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
  #slider { flex: 1 1 260px; min-width: 180px; accent-color: var(--accent); }
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
</style>
</head>
<body>
<header>
  <div class="top">
    <div>
      <h1>SolarCast <span>NZ</span></h1>
      <div class="tagline">How much power your solar panels will give next hour, anywhere in New Zealand</div>
    </div>
    <div class="clock">
      <div class="nz" id="nzTime">--</div>
      <div class="mode" id="modeLabel">Loading...</div>
    </div>
  </div>
  <div class="controls">
    <label for="slider">Replay</label>
    <input type="range" id="slider" min="0" max="0" value="0" aria-label="Replay time">
    <button id="liveBtn" type="button">Live</button>
    <label for="sizeSel">Your panels</label>
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
      <select id="colorMode"><option value="power">Predicted next-hour power (your panels)</option><option value="accuracy">Model accuracy (error)</option></select>
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
    <div class="card wide"><h3>How good is our model?</h3><div class="sub" id="mPeriods">Loading...</div></div>
    <div class="card wide"><h3 id="hVersion">Model version</h3><div id="mVersion" class="sub">Loading...</div></div>
    <div class="card"><h3 id="hOverallKw">Overall error (kW)</h3><div class="sub">Mean absolute error, lower is better</div><div class="chartbox"><canvas id="cOverallKw"></canvas></div></div>
    <div class="card"><h3 id="hOverallWm">Overall error (W/m&sup2;)</h3><div class="sub">Mean absolute error of sunlight, lower is better</div><div class="chartbox"><canvas id="cOverallWm"></canvas></div></div>
    <div class="card wide"><h3>All overall metrics</h3><div id="mTable"></div></div>
    <div class="card wide"><h3>Error per region (MAE, kW)</h3><div class="sub">Test period, lower is better</div><div class="chartbox tall"><canvas id="cRegion"></canvas></div></div>
    <div class="card wide"><h3 id="hSample">Sample week</h3><div class="sub">Predicted vs actual power for one week</div><div class="chartbox tall"><canvas id="cSample"></canvas></div></div>
    <div class="card wide"><h3>What the model looks at (top feature importances)</h3><div class="chartbox tall"><canvas id="cImp"></canvas></div></div>
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
    <button type="button">Should the workshop charge its EV now?</button>
    <button type="button">我的太陽能板今天在奧克蘭能發多少電？</button>
  </div>
  <div id="chatCtx"></div>
  <form class="cform" id="chatForm"><input id="chatInput" maxlength="500" autocomplete="off" placeholder="Ask about this data..." aria-label="Your question"><button id="chatSend" type="submit">Send</button></form>
</section>

<footer>
  AI chat: Qwen3 (Apache-2.0) running locally on the notebook's machine, safety check by Llama Guard 3 (Built with Llama); it answers only from this dashboard's data and may make mistakes.
  Weather: Open-Meteo (CC BY 4.0), reanalysis/forecast model data. PV output derived from irradiance and scaled to the panel size you pick. Prototype; not energy or financial advice.
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
const NZ_FMT = new Intl.DateTimeFormat('en-NZ', { timeZone: 'Pacific/Auckland', weekday: 'short', day: 'numeric', month: 'short',
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
    ? `<b class="gen">Your ${SIZE} kW panels: ${num(S(r.predicted_kw_next), kwd())} kW expected next hour</b>`
    : `<b class="nogen">Not generating${r.is_day ? ' (too cloudy)' : ' (night)'}: ${num(S(r.predicted_kw_next), kwd())} kW expected</b>`;
  const acc = (regionMae[r.name] !== undefined && $('colorMode').value === 'accuracy')
    ? `Model error here: ${num(S(regionMae[r.name]), 2)} kW (MAE, your ${SIZE} kW system)<br>` : '';
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
    $('legend').innerHTML = `<b>Model error (MAE, kW, ${SIZE} kW system)</b>` + (accBins.length
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
  if (current === 'live' && document.visibilityState === 'visible') loadSnapshot({ panel: false });
}, 60000);

// ---------- header: clock, mode, note ----------
function setNote(text, kind) {
  const n = $('note');
  n.textContent = text || '';
  n.className = 'note' + (kind ? ' ' + kind : '');
  n.style.display = text ? '' : 'none';
}

function updateHeader(regions) {
  const local = regions.length ? regions[0].local_time : nzString(current === 'live' ? new Date().toISOString() : current);
  $('nzTime').innerHTML = esc(local) + ' <small>New Zealand time</small>';
  $('modeLabel').textContent = current === 'live' ? 'Live mode (current hour, recent data and forecast)' : 'Replay of a past hour';
  $('liveBtn').classList.toggle('on', current === 'live');
  const anyDay = regions.some(r => r.is_day);
  const anyGen = regions.some(r => r.generating);
  if (!regions.length) setNote('');
  else if (!anyDay) setNote('It is night in New Zealand right now, so solar output is zero everywhere. Judges in the US often see NZ at night in Live mode: drag the Replay slider to a New Zealand daytime hour to see the panels generate.', 'night');
  else if (!anyGen) setNote('Solar output is near zero in every region at this hour. Try another time with the Replay slider.', 'night');
  else setNote('At night solar power is zero. Use the Replay slider to move through New Zealand days, or press Live for the current hour.');
}

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
    updateHeader(data.regions);
  } catch (e) {
    if (id === reqId) setNote('Could not load data: ' + e.message, 'error');
  }
  if (opts.panel === false) return;
  if (selected) loadDetail(selected);
  else if (selectedPoint) loadPoint();
}

function setTime(t) {
  current = t;
  if (t === 'live') { $('slider').dataset.live = '1'; }
  else {
    delete $('slider').dataset.live;
    const i = times.indexOf(t);
    if (i >= 0) $('slider').value = i;
  }
  $('nzTime').innerHTML = esc(t === 'live' ? 'Live' : nzString(t)) + ' <small>New Zealand time</small>';
  loadSnapshot();
}

$('liveBtn').onclick = () => setTime('live');
$('slider').addEventListener('input', () => {
  const t = times[+$('slider').value];
  $('nzTime').innerHTML = esc(nzString(t)) + ' <small>New Zealand time</small>';
  clearTimeout(debounce);
  debounce = setTimeout(() => setTime(t), 150);
});

// ---------- detail panel ----------
function hourLabel(h) { return (typeof h === 'number') ? String(h).padStart(2, '0') + ':00' : String(h); }

function metricsHtml(m, label) {
  if (!m || typeof m !== 'object') return '';
  const rows = Object.entries(m).filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, ' '))}</td><td>${num(/_kw$/.test(k) ? S(v) : v, 2)}</td></tr>`).join('');
  return rows ? `<h3>${label || 'Model accuracy in this region (test period)'}</h3><table class="m">${rows}</table>` +
    `<div class="sub">kW errors are scaled to your ${SIZE} kW system (the error is the same share of capacity at any size).</div>` : '';
}

async function loadDetail(name) {
  const id = ++detailId;
  const panel = $('panel');
  if (!panel.querySelector('canvas')) panel.innerHTML = `<div class="empty">Loading ${esc(name)}...</div>`;
  let d;
  try { d = await getJson('/api/detail?region=' + encodeURIComponent(name) + '&t=' + encodeURIComponent(current)); }
  catch (e) { if (id === detailId) panel.innerHTML = `<div class="empty">Could not load ${esc(name)}: ${esc(e.message)}</div>`; return; }
  if (id !== detailId) return;
  renderPanel(d, esc(name), `Local day, hourly power of your ${SIZE} kW rooftop system`);
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
  const gen = d.generating ? `<b style="color:var(--good)">Your ${SIZE} kW panels: ${num(S(d.predicted_kw_next), kwd())} kW expected next hour</b>`
                           : `<b>Not generating: ${num(S(d.predicted_kw_next), kwd())} kW expected next hour</b>`;
  pin.setTooltipContent(`<b>${num(d.lat, 2)}, ${num(d.lon, 2)}</b><br>${gen}`);
  renderPanel(d, `Point ${num(d.lat, 2)}, ${num(d.lon, 2)}`,
    `${esc(d.local_time)} NZ &middot; ${gen}<br>Cloud ${num(d.cloud_now_pct, 0)}% &middot; ${num(d.temp_c)} &deg;C &middot; wind ${num(d.wind_kmh, 0)} km/h &middot; sunlight now ${num(d.ghi_now_wm2, 0)} W/m&sup2;<br>` +
    `Weather downloaded for this exact spot; the model uses the climate of the nearest region, <b>${esc(d.nearest_region)}</b> (${num(d.distance_km, 0)} km away).`);
}
function selectPoint(lat, lon) {
  selected = null; selectedPoint = { lat: +lat.toFixed(4), lon: +lon.toFixed(4) };
  Object.values(markers).forEach(m => m.setStyle({ radius: 14, color: '#ffffff', weight: 2 }));
  if (!pin) pin = L.circleMarker([lat, lon], { radius: 9, color: '#ffffff', weight: 3, fillColor: '#2563eb', fillOpacity: 1, bubblingMouseEvents: false })
    .addTo(map).bindTooltip('', { direction: 'top', offset: [0, -8], className: 'tip' });
  pin.setLatLng([lat, lon]);
  loadPoint();
  if (window.matchMedia('(max-width: 900px)').matches) $('panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
map.on('click', e => selectPoint(e.latlng.lat, e.latlng.lng));

let lastPanel = null;
function yoursHtml(d) {
  const sum = a => (a || []).reduce((t, v) => t + (typeof v === 'number' ? v : 0), 0);
  const day = S(sum(d.predicted_kw)), done = (d.actual_kw || []).some(v => typeof v === 'number') ? S(sum(d.actual_kw)) : null;
  return `<div class="yours"><div class="sub">Your ${SIZE} kW panels, next hour</div>
    <div class="big">${num(S(d.predicted_kw_next), kwd())} kW <small>&asymp; ${num(S(d.predicted_kw_next), kwd())} kWh over the hour</small></div>
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
  const mlabel = d.is_point ? `Model accuracy in the nearest region (${esc(d.nearest_region)}, test period)` : null;
  panel.innerHTML = `
    <h2>${titleHtml}</h2>
    <div class="sub">${subHtml}</div>
    ${yoursHtml(d)}
    <div class="chartbox"><canvas id="chart"></canvas></div>
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
}

function drawChart(d) {
  if (chart) { chart.destroy(); chart = null; }
  const labels = (d.hours || []).map(hourLabel);
  chart = new Chart($('chart'), {
    data: {
      labels,
      datasets: [
        { type: 'line', label: 'Actual (derived from weather data)', data: (d.actual_kw || []).map(S), borderColor: '#1f8f4d', backgroundColor: '#1f8f4d', borderWidth: 2.5, pointRadius: 0, tension: 0.25, spanGaps: false, order: 1 },
        { type: 'line', label: 'Predicted (model)', data: (d.predicted_kw || []).map(S), borderColor: '#e8890c', backgroundColor: '#e8890c', borderWidth: 2.5, borderDash: [6, 4], pointRadius: 0, tension: 0.25, order: 2 },
        { type: 'line', label: 'Persistence (same as last hour)', data: (d.persistence_kw || []).map(S), borderColor: '#6b7684', backgroundColor: '#6b7684', borderWidth: 1.5, borderDash: [2, 3], pointRadius: 0, tension: 0.25, order: 3 },
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

function selectRegion(name) {
  selected = name; selectedPoint = null;
  if (pin) { map.removeLayer(pin); pin = null; }
  Object.entries(markers).forEach(([n, m]) => m.setStyle({ radius: n === name ? 17 : 14, color: n === name ? '#1c2733' : '#ffffff', weight: n === name ? 3 : 2 }));
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
function mchart(id, config) {
  if (mcharts[id]) mcharts[id].destroy();
  mcharts[id] = new Chart($(id), config);
}
function renderModel() {
  if (!report) { $('mPeriods').textContent = 'Model report not available.'; return; }
  const p = report.periods || {};
  $('mPeriods').innerHTML = Object.entries(p).map(([k, v]) =>
    `<b>${esc(k)}</b>: ${esc(Array.isArray(v) ? v.join(' to ') : v)}`).join(' &nbsp;|&nbsp; ') +
    `<br>kW numbers on this tab are for the ${REF} kW reference system. Errors scale with panel size: divide by ${REF / SIZE} for your ${SIZE} kW system (2.2% of capacity at any size).`;

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
  const names = overall.map(o => o.model);
  const colors = names.map((_, i) => MODEL_COLORS[i % MODEL_COLORS.length]);
  const bar = (id, key, title) => mchart(id, { type: 'bar',
    data: { labels: names, datasets: [{ label: title, data: overall.map(o => o[key]), backgroundColor: colors }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } } });
  bar('cOverallKw', 'mae_kw', 'MAE (kW)');
  bar('cOverallWm', 'mae_wm2', 'MAE (W/m2)');
  const cols = [['mae_kw', 'MAE kW'], ['mae_wm2', 'MAE W/m2'], ['rmse_kw', 'RMSE kW']];
  $('mTable').innerHTML = '<table class="m"><tr><th>Model</th>' + cols.map(c => `<th>${c[1]}</th>`).join('') + '</tr>' +
    overall.map(o => `<tr><td>${esc(o.model)}</td>` + cols.map(c => `<td>${num(o[c[0]], 2)}</td>`).join('') + '</tr>').join('') + '</table>';

  const rows = report.per_region || [];
  const regions = [...new Set(rows.map(r => r.region))];
  const models = [...new Set(rows.map(r => r.model))];
  mchart('cRegion', { type: 'bar',
    data: { labels: regions, datasets: models.map((m, i) => ({ label: m, backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length],
      data: regions.map(rg => { const r = rows.find(x => x.region === rg && x.model === m); return r ? r.mae_kw : null; }) })) },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'MAE (kW)' } }, x: { ticks: { maxRotation: 60, minRotation: 40 } } } } });

  const s = report.sample || {};
  $('hSample').textContent = 'Sample week' + (s.region ? ': ' + s.region : '');
  mchart('cSample', { type: 'line',
    data: { labels: (s.times || []).map(nzString), datasets: [
      { label: 'Actual', data: s.actual_kw || [], borderColor: '#1f8f4d', borderWidth: 2, pointRadius: 0, tension: 0.2 },
      { label: 'Predicted (model)', data: s.predicted_kw || [], borderColor: '#e8890c', borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: 0.2 },
      { label: 'Persistence', data: s.persistence_kw || [], borderColor: '#6b7684', borderWidth: 1.2, borderDash: [2, 3], pointRadius: 0, tension: 0.2 } ] },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'bottom' } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'kW' } }, x: { ticks: { maxTicksLimit: 8 } } } } });

  const imp = [...(report.importance || [])].sort((a, b) => b.importance - a.importance).slice(0, 10);
  mchart('cImp', { type: 'bar',
    data: { labels: imp.map(i => i.feature), datasets: [{ label: 'Importance', data: imp.map(i => i.importance), backgroundColor: '#e8890c' }] },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } } });
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
  const when = current === 'live' ? 'live data for the current hour'
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
    $('slider').max = Math.max(0, times.length - 1);
    const i = times.indexOf(info.default);
    $('slider').value = i >= 0 ? i : 0;              // the Replay slider starts at a sunny past midday
    setTime('live');                                 // but the page always opens on the current hour (Live)
  } catch (e) { setNote('Could not load the time list: ' + e.message, 'error'); }
})();
</script>
</body>
</html>
"""
