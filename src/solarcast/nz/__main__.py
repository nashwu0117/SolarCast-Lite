"""Run the SolarCast NZ dashboard on your own computer.

    pip install -e .
    python -m solarcast.nz                  # http://127.0.0.1:8000  (map, nationwide grid, chat with templates)
    python -m solarcast.nz --ai             # also use a local Ollama model for the chat and the briefing
    python -m solarcast.nz --tunnel         # also open a public Cloudflare quick-tunnel link

--ai needs Ollama (https://ollama.com) running on this computer with the model pulled first, for example
`ollama pull qwen3:1.7b` (any computer) or `ollama pull qwen3:8b` (a GPU or Apple silicon with 16 GB).
Without it the chat still answers with fixed templates built from the same data.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from solarcast.nz import app, webapp
from solarcast.nz.assistant import Assistant, OllamaLLM

OLLAMA = "http://127.0.0.1:11434"


def _local_llm(model: str):
    """An OllamaLLM if Ollama is running and has `model`; otherwise None (templates)."""
    try:
        tags = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3).read().decode())
    except Exception:
        print(f"Ollama is not running at {OLLAMA}: the chat will use templates. Start it with `ollama serve`.")
        return None
    names = {m.get("name") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        print(f"Ollama does not have {model}: run `ollama pull {model}`. The chat will use templates.")
        return None
    return OllamaLLM(model, OLLAMA, timeout_s=300, num_ctx=8192, max_tokens=300)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m solarcast.nz", description="SolarCast NZ dashboard")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tunnel", action="store_true", help="open a public Cloudflare quick-tunnel link")
    p.add_argument("--ai", action="store_true", help="use a local Ollama model for the chat and briefing")
    p.add_argument("--model", default="qwen3:1.7b", help="Ollama model for --ai (default qwen3:1.7b)")
    p.add_argument("--no-grid", action="store_true", help="skip the nationwide grid (no extra downloads)")
    args = p.parse_args(argv)

    print("Loading the model and the replay data ...")
    svc = app.load_service()
    llm = _local_llm(args.model) if args.ai else None
    grid = None
    if not args.no_grid:
        from solarcast.nz.grid import NationalGrid
        grid = NationalGrid(svc)
        print("Computing the nationwide grid (about 30-60 s) ...", grid.precompute(svc.default_time()))
        grid.start_hourly()
    assistant = Assistant(svc, llm, grid=grid)
    if grid is not None:
        grid.summarizer = assistant.summarize
    url = webapp.serve(svc, port=args.port, tunnel=args.tunnel, assistant=assistant, grid=grid)
    print(f"\nSolarCast NZ is running: {url}\nPress Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
