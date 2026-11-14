"""Which local language model should the judges' T4 run for SolarCast AI: Qwen3 8B or Qwen3 14B?

Same dashboard data, same questions, two rounds per model. Every answer is checked automatically against the FACTS the
server gives the model (the right kW / kWh / NZD for the chosen panel size, the right region, the EV advice, the
accuracy, replay wording, Traditional Chinese characters). Also measured: setup time (download + load), GPU memory with
the model and the Llama Guard 3 1B safety model loaded, and answer time.

Run on Colab with a T4:  PYTHONPATH=src python experiments/nz_llm_compare.py [qwen3:8b qwen3:14b]
-> experiments/results/nz_llm_compare.{json,md}
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarcast.nz import app, llm_setup                 # noqa: E402
from solarcast.nz.assistant import Assistant             # noqa: E402

OUT = ROOT / "experiments" / "results"
SIMPLIFIED = set("电发时预计这个们说网对应么为气东区间较总")      # characters that only appear in Simplified Chinese


def nums(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]


def has(text: str, value, tol: float) -> bool:
    return value is not None and any(abs(x - value) <= tol for x in nums(text))


def questions(a: Assistant, t: str):
    """(question, region for the panel, system_kw, check(answer) -> list of failed checks)."""
    f5 = a.facts(t, "Christchurch", system_kw=5)
    s5 = f5["selected_region"]
    f10 = a.facts(t, "Auckland", system_kw=10)
    s10 = f10["selected_region"]
    top = a.facts(t, None, system_kw=5)["regions_now"][0]
    ev = a.facts(t, "Christchurch", system_kw=50)["ev_example_workshop"]["advice_next_hour"]
    acc = f5["model"]["test_period_results_daytime"]["our_model"]
    replay = f5["dashboard_time"]["mode"].startswith("REPLAY")
    ref50 = a.facts(t, "Christchurch")["selected_region"]["predicted_kw_next_hour"]

    def c1(ans):
        return [m for ok, m in [(has(ans, s5["predicted_kw_next_hour"], 0.06), f"next hour {s5['predicted_kw_next_hour']} kW"),
                                (has(ans, s5["day_predicted_kwh"], 0.6), f"day {s5['day_predicted_kwh']} kWh"),
                                (not has(ans, ref50, 0.06), f"must not quote the 50 kW figure {ref50}")] if not ok]

    def c2(ans):
        bad = [m for ok, m in [(has(ans, s10["predicted_kw_next_hour"], 0.06), f"next hour {s10['predicted_kw_next_hour']} kW"),
                               (has(ans, s10["day_value_nzd_at_grid_price"], 0.06), f"NZD {s10['day_value_nzd_at_grid_price']}")]
               if not ok]
        if SIMPLIFIED & set(ans):
            bad.append("Simplified characters: " + "".join(sorted(SIMPLIFIED & set(ans))))
        if not re.search(r"[一-鿿]", ans):
            bad.append("not in Chinese")
        return bad

    def c3(ans):
        return [m for ok, m in [(top["region"].lower() in ans.lower(), f"region {top['region']}"),
                                (has(ans, top["predicted_kw_next_hour"], 0.06), f"{top['predicted_kw_next_hour']} kW")] if not ok]

    def c4(ans):
        yes = bool(re.search(r"\b(yes|charge now|should charge)\b", ans, re.I)) and not re.search(r"\b(no,|should not|shouldn't|wait)\b", ans, re.I)
        return [] if yes == bool(ev["charge_now"]) else [f"EV advice should be {'charge' if ev['charge_now'] else 'wait'}"]

    def c5(ans):
        return [] if (has(ans, acc["accuracy_pct"], 0.06) or has(ans, acc["error_pct_of_capacity"], 0.06)) else \
            [f"accuracy {acc['accuracy_pct']}% / error {acc['error_pct_of_capacity']}% of capacity"]

    def c6(ans):
        return [] if not replay or re.search(r"replay|past|not (the current|now)|回放|過去", ans, re.I) else \
            ["should say the shown hour is a replay, not now"]

    return [
        ("I have 5 kW of panels in Christchurch. How much power will they give next hour, and how much today?", "Christchurch", 5, c1),
        ("我在奧克蘭有 10 kW 的太陽能板，下一小時能發多少電？今天的發電大概值多少錢？", "Auckland", 10, c2),
        ("Which region will give the most solar power next hour?", None, 5, c3),
        ("Should the Christchurch workshop charge its EV now?", "Christchurch", 50, c4),
        ("How accurate is the forecasting model?", None, 5, c5),
        ("Is it sunny in Wellington right now?", "Wellington", 5, c6),
    ]


def gpu_mem_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else -1


def unload(model: str):
    try:
        req = urllib.request.Request(llm_setup.HOST + "/api/generate", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": model, "prompt": "", "keep_alive": 0}).encode())
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def main(models):
    svc = app.load_service()
    t = svc.default_time()
    results = {}
    for model in models:
        start = time.time()
        llm = llm_setup.start(model)
        while llm.status()["state"] not in ("ready", "templates"):
            time.sleep(3)
        setup_s = round(time.time() - start)
        if llm.status()["state"] != "ready":
            results[model] = {"error": llm.status()["detail"]}
            continue
        a = Assistant(svc, llm)
        qs = questions(a, t)
        rows = []
        for rnd in (1, 2):
            for q, region, size, check in qs:
                t0 = time.time()
                res = a.ask(q, t, region=region, system_kw=size)
                fails = check(res["answer"]) if res["mode"] == "llm" else [f"mode {res['mode']}"]
                rows.append({"round": rnd, "q": q, "size_kw": size, "mode": res["mode"], "seconds": round(time.time() - t0, 1),
                             "answer": res["answer"], "failed": fails})
                print(f"[{model} r{rnd}] {'OK ' if not fails else 'BAD'} {rows[-1]['seconds']:5.1f}s  {q[:50]}  {fails}", flush=True)
        passed = sum(not r["failed"] for r in rows)
        results[model] = {"setup_seconds": setup_s, "gpu_memory_mb": gpu_mem_mb(), "passed": passed, "total": len(rows),
                          "median_answer_s": sorted(r["seconds"] for r in rows)[len(rows) // 2], "answers": rows}
        print(f"== {model}: {passed}/{len(rows)} checks passed, setup {setup_s} s, GPU {results[model]['gpu_memory_mb']} MB, "
              f"median answer {results[model]['median_answer_s']} s", flush=True)
        unload(model)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nz_llm_compare.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = ["# SolarCast AI: Qwen3 8B vs 14B on a Colab T4 (same data and questions, two rounds)", "",
          "| Model | Checks passed | Setup (download + load) | GPU memory (with Llama Guard 1B) | Median answer |",
          "|---|---|---|---|---|"]
    md += [f"| {m} | {r['passed']}/{r['total']} | {r['setup_seconds']} s | {r['gpu_memory_mb']} MB | {r['median_answer_s']} s |"
           for m, r in results.items() if "passed" in r]
    md += ["", "Failed checks:", ""]
    md += [f"- {m} round {x['round']}: \"{x['q']}\" -> {', '.join(x['failed'])}"
           for m, r in results.items() if "answers" in r for x in r["answers"] if x["failed"]]
    (OUT / "nz_llm_compare.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3:8b", "qwen3:14b"])
