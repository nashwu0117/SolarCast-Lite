"""Start a local language model for SolarCast AI on the Colab VM, in the background.

    llm = llm_setup.start()            # returns at once; the download continues in a thread
    llm.status()                       # {"state": "downloading", "detail": "...", "model": ...}

- With an NVIDIA GPU of 14 GB or more (Colab T4, 15 GB): Qwen3 14B (about 9 GB, 4-bit) through Ollama; on a smaller GPU
  Qwen3 8B (about 5 GB). experiments/nz_llm_compare.py: both pass every check, 14B explains the numbers more precisely;
  it answers in about 10 s instead of 4.5 s and uses 11.7 GB of GPU memory with the safety model.
- Without a GPU: the small Qwen3 1.7B on the CPU (slower answers), so the chat still works.
- If anything fails, the state becomes "templates" and the dashboard answers with fixed templates.

Ollama is installed from its official install script and serves on 127.0.0.1 only (not exposed through the tunnel).
The model gets no tools: it cannot browse or run code (see assistant.py).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from solarcast.nz.assistant import OllamaLLM

GPU_MODEL = os.environ.get("SOLARCAST_LLM_GPU", "qwen3:14b")         # GPUs with >= BIG_GPU_MB of memory
SMALL_GPU_MODEL = os.environ.get("SOLARCAST_LLM_SMALL_GPU", "qwen3:8b")
BIG_GPU_MB = 14000
CPU_MODEL = os.environ.get("SOLARCAST_LLM_CPU", "qwen3:1.7b")
GUARD_MODEL = os.environ.get("SOLARCAST_LLM_GUARD", "llama-guard3:1b")
HOST = "http://127.0.0.1:11434"
INSTALL_URL = "https://ollama.com/install.sh"


def gpu_memory_mb() -> int:
    """Total memory of the first NVIDIA GPU in MB (0 when there is none or nvidia-smi fails)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def has_gpu() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class LocalLLM(OllamaLLM):
    """OllamaLLM plus a background setup thread and a status() the dashboard can show."""

    def __init__(self, model: str, guard_model: str | None, gpu: bool):
        # A 2-core CPU reads the facts slowly: smaller context, shorter replies, longer timeout.
        super().__init__(model, HOST, guard_model=None, timeout_s=120 if gpu else 300,
                         num_ctx=8192 if gpu else 4096, max_tokens=450 if gpu else 160)
        self._guard_wanted, self.gpu = guard_model, gpu
        self._state = {"state": "starting", "detail": "Preparing the AI model...", "model": self.name,
                       "device": "GPU (NVIDIA)" if gpu else "CPU"}
        self.log: list[str] = []

    def status(self) -> dict:
        return dict(self._state)

    def _set(self, state: str, detail: str):
        self._state.update(state=state, detail=detail)
        self.log.append(f"{time.strftime('%H:%M:%S')} {state}: {detail}")

    # ----- setup steps -----
    def _sh(self, cmd: str, timeout: int = 900):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"`{cmd.split()[0]}` failed: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout

    def _server_up(self) -> bool:
        try:
            urllib.request.urlopen(HOST + "/api/version", timeout=3).read()
            return True
        except Exception:
            return False

    def _pull(self, model: str):
        """Stream /api/pull and keep a progress percentage in the status."""
        req = urllib.request.Request(HOST + "/api/pull", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as res:
            for line in res:
                ev = json.loads(line.decode() or "{}")
                if ev.get("error"):
                    raise RuntimeError(ev["error"])
                if ev.get("total") and ev.get("completed"):
                    pct = ev["completed"] / ev["total"] * 100
                    self._state["detail"] = f"Downloading {model}: {pct:.0f}% of {ev['total'] / 1e9:.1f} GB"

    def setup(self):
        try:
            if not install_complete():                            # missing, or cut off by a restart mid-install
                self._sh("pkill -x ollama || true")
                self._set("installing", "Installing Ollama (local model server)...")
                self._sh("command -v zstd >/dev/null || (apt-get -qq update && apt-get -qq install -y zstd) >/dev/null 2>&1")
                self._sh(f"curl -fsSL {INSTALL_URL} | sh")
            if not self._server_up():
                env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_KEEP_ALIVE": "-1"}
                subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(60):
                    if self._server_up():
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("the Ollama server did not start")
            self._set("downloading", f"Downloading {self.model}...")
            self._pull(self.model)
            if self._guard_wanted:
                try:
                    self._set("downloading", f"Downloading the safety classifier {self._guard_wanted}...")
                    self._pull(self._guard_wanted)
                    self.guard_model = self._guard_wanted
                except Exception as exc:                          # the keyword filter still runs without it
                    self.log.append(f"guard model skipped: {exc}")
            self._warm(self.model)
            if self.guard_model:
                try:
                    self._warm(self.guard_model)
                except Exception as exc:
                    self.log.append(f"guard model skipped: {exc}")
                    self.guard_model = None
            self._set("ready", f"{self.model} ready on {self._state['device']}")
        except Exception as exc:
            self._set("templates", f"AI model unavailable during '{self._state['state']}' ({exc}); "
                                   "answering with fixed templates.")

    def _warm(self, model: str, tries: int = 3):
        """Load a model into (GPU) memory. The first load of a fresh 5 GB download can take minutes."""
        self._set("warming", f"Loading {model} into memory (the first time can take a few minutes)...")
        for attempt in range(tries):
            try:
                req = urllib.request.Request(HOST + "/api/generate", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"model": model, "prompt": "", "keep_alive": -1}).encode())
                urllib.request.urlopen(req, timeout=600).read()
                return
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(5)


def install_complete() -> bool:
    """True when Ollama is installed with its model engine. If a session restarts while the installer is still
    unpacking, `ollama` exists but lib/ollama/llama-server does not, and every model load fails; then reinstall."""
    exe = shutil.which("ollama")
    if not exe:
        return False
    lib = Path(exe).resolve().parent.parent / "lib" / "ollama"
    return (lib / "llama-server").exists()


def choose_model(gpu: bool, memory_mb: int) -> str:
    """14B on a GPU with enough memory (a Colab T4 has 15 GB), 8B on a smaller GPU, 1.7B on the CPU."""
    if not gpu:
        return CPU_MODEL
    return GPU_MODEL if memory_mb >= BIG_GPU_MB else SMALL_GPU_MODEL


def start(model: str | None = None, guard: bool = True) -> LocalLLM:
    """Start the background setup and return the LLM object at once."""
    gpu = has_gpu()
    # On a CPU the extra safety model would double the wait; the keyword filter and the rules still apply.
    llm = LocalLLM(model or choose_model(gpu, gpu_memory_mb() if gpu else 0), GUARD_MODEL if guard and gpu else None, gpu)
    threading.Thread(target=llm.setup, name="solarcast-llm-setup", daemon=True).start()
    return llm
