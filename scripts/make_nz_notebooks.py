"""Generate the four SolarCast NZ notebooks from ONE source so English and zh-TW never drift.

    python scripts/make_nz_notebooks.py

Writes notebooks/train.ipynb, notebooks/train.zh-TW.ipynb, notebooks/SolarCast_NZ_Judges.ipynb and
notebooks/SolarCast_NZ_Judges.zh-TW.ipynb. Code cells are identical in both languages except for the comments:
a code line `# @@key@@` is replaced by the comment text CODE_COMMENTS[key][lang] (keeping the indentation).
Markdown cells are written per language in MD[...].
"""
import json
import re
import uuid
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
MODEL_JSON = json.loads((Path(__file__).resolve().parents[1] / "models" / "nz_model.json").read_text())
SK = MODEL_JSON["sklearn_version"]
REPO_URL = "https://github.com/nashwu0117/SolarCast-Lite"
LANGS = ("en", "zh-TW")

# ---------------------------------------------------------------------------------------------------------------
# Code comments (the only language-dependent part of the code cells)
# ---------------------------------------------------------------------------------------------------------------
CODE_COMMENTS = {
    "clone": {"en": "Get the code from the public GitHub repository (shallow clone: fast).",
              "zh-TW": "從公開的 GitHub 儲存庫取得程式碼（淺層 clone，速度很快）。"},
    "sk": {"en": "The saved model needs the same scikit-learn version it was trained with.",
           "zh-TW": "存好的模型必須搭配訓練時相同版本的 scikit-learn 才能載入。"},
    "path": {"en": "Make `import solarcast` work.", "zh-TW": "讓 `import solarcast` 可以運作。"},
    "load": {"en": "Load the trained model and the replay weather (about 15 months of Open-Meteo data).",
             "zh-TW": "載入訓練好的模型與回放用的天氣資料（約 15 個月的 Open-Meteo 資料）。"},
    "acc": {"en": "Accuracy as a share of the 50 kW system: 100 - nMAE, where nMAE = MAE / 50 kW.",
            "zh-TW": "以 50 kW 系統為基準的準確度：100 - nMAE，其中 nMAE = MAE / 50 kW。"},
    "overall": {"en": "Overall TEST results (daytime hours, July 2025 - Oct 2026), saved at training time.",
                "zh-TW": "整體測試集結果（白天時段，2025 年 7 月至 2026 年 10 月），訓練時就已存好。"},
    "perregion": {"en": "Per-region TEST MAE in kW (lower is better).",
                  "zh-TW": "各區域的測試集 MAE（單位 kW，越小越好）。"},
    "curve": {"en": "Learning curve: how the error falls as the model sees more training days.",
              "zh-TW": "學習曲線：模型看過越多訓練天數，誤差就越小。"},
    "sample": {"en": "One sample week: actual power vs. the model vs. the persistence baseline.",
               "zh-TW": "取樣一週：實際發電量、模型預測、以及「持續性」基準線的比較。"},
    "serve": {"en": "Start the dashboard (Flask) and open a free Cloudflare tunnel so anyone can reach it.",
              "zh-TW": "啟動儀表板（Flask），並開啟免費的 Cloudflare 通道，讓任何人都能連進來。"},
    "big": {"en": "Show the link big. If there is no tunnel, the local URL is shown instead.",
            "zh-TW": "把連結用大字顯示。如果沒有通道，就會顯示本機網址。"},
    # ----- train notebook -----
    "quick": {"en": "QUICK = True runs a fast smoke test: 2 regions x 120 days (the repo model is NOT touched).",
              "zh-TW": "QUICK = True 會跑快速冒煙測試：2 個區域 × 120 天（不會動到儲存庫裡的模型）。"},
    "quick_cfg": {"en": "Small windows: 60 % train, 20 % validation, 20 % test, in time order.",
                  "zh-TW": "縮小的時間窗：60% 訓練、20% 驗證、20% 測試，依時間先後切分。"},
    "full_cfg": {"en": "Full run: all 14 regions, 2022 to about a week ago.",
                 "zh-TW": "完整執行：14 個區域，從 2022 年到大約一週前。"},
    "fetch": {"en": "Download hourly history from Open-Meteo (cached on disk, retries on rate limits).",
              "zh-TW": "從 Open-Meteo 下載每小時歷史資料（會存在硬碟快取，遇到流量限制會自動重試）。"},
    "feat": {"en": "Features use only information known at the end of the issue hour (no leakage).",
             "zh-TW": "特徵只使用「發出預測那一刻」已知的資訊（沒有偷看未來）。"},
    "baseline": {"en": "Two baselines every model has to beat: persistence and clear-sky persistence.",
                 "zh-TW": "任何模型都必須打敗的兩個基準線：持續性預測與晴空持續性預測。"},
    "tune": {"en": "Tune on the VALIDATION period only; the test period is not touched here.",
             "zh-TW": "只用「驗證集」調參數；這裡完全沒有碰測試集。"},
    "test": {"en": "The ONE look at the test period, with the chosen model.",
             "zh-TW": "只看「一次」測試集，並且只用選定的那個模型。"},
    "lc_old": {"en": "The learning curve was computed earlier and is stored in the JSON of the committed model.",
               "zh-TW": "學習曲線先前已經算好，存在儲存庫裡已提交模型的 JSON 檔中。"},
    "save": {"en": "Save the model. QUICK runs write to /content so the repo model stays untouched.",
             "zh-TW": "儲存模型。QUICK 模式會寫到 /content，所以儲存庫裡的模型不會被覆蓋。"},
    "backup": {"en": "Set BACKUP = True to commit and push the model to GitHub (needs the Colab Secret GITHUB_TOKEN).",
               "zh-TW": "把 BACKUP 改成 True，就會把模型 commit 並 push 到 GitHub（需要 Colab Secret：GITHUB_TOKEN）。"},
    "token": {"en": "The token is read from Colab Secrets and is never printed (it is masked in all output).",
              "zh-TW": "token 從 Colab Secrets 讀取，而且絕不會印出來（所有輸出都會把它遮成 ***）。"},
}


def comment(match: re.Match, lang: str) -> str:
    return f"{match.group(1)}# {CODE_COMMENTS[match.group(2)][lang]}"


def code(src: str, lang: str) -> str:
    return re.sub(r"^([ \t]*)# @@(\w+)@@", lambda m: comment(m, lang), src.strip("\n"), flags=re.M)


# ---------------------------------------------------------------------------------------------------------------
# Code cells shared by the notebooks
# ---------------------------------------------------------------------------------------------------------------
JUDGES_SETUP = f'''
import importlib.metadata as md
import json, os, subprocess, sys

REPO = "/content/SolarCast-Lite"
# @@clone@@
if not os.path.isdir(f"{{REPO}}/src"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}.git", REPO], check=True)

# @@sk@@
want = json.load(open(f"{{REPO}}/models/nz_model.json"))["sklearn_version"]
need = ["flask"] + ([] if md.version("scikit-learn") == want else [f"scikit-learn=={{want}}"])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
if "sklearn" in sys.modules and sys.modules["sklearn"].__version__ != want:
    raise SystemExit(f"scikit-learn {{want}} was just installed: use Runtime > Restart session, then Run all again.")

# @@path@@
sys.path.insert(0, f"{{REPO}}/src")
os.chdir(REPO)
print("ready; scikit-learn", md.version("scikit-learn"))
'''

JUDGES_LOAD = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from solarcast.nz import app, config, model as nzmodel

# @@load@@
svc = app.load_service()
meta = svc.meta
test = meta["metrics"]["test"]
print(f"{len(svc.regions)} regions | replay hours: {len(svc.available_times())} | "
      f"model trained on {meta['periods']['train'][0]} to {meta['periods']['train'][1]} | data: {meta['data_source']}")
'''

JUDGES_OVERALL = '''
# @@acc@@
def accuracy(mae_kw):
    return 100 - mae_kw / config.SYSTEM_KW * 100

# @@overall@@
rows = []
for key, m in test["overall"].items():
    rows.append({"model": nzmodel.MODEL_NAMES[key], "MAE (kW)": m["mae_kw"], "MAE (W/m2)": m["mae_wm2"],
                 "RMSE (kW)": m["rmse_kw"], "accuracy (% of 50 kW)": accuracy(m["mae_kw"]), "hours": m["n"]})
overall = pd.DataFrame(rows).round(2)
display(overall)
best, pers = test["overall"]["model"]["mae_kw"], test["overall"]["persistence"]["mae_kw"]
print(f"The model's error is {100 - best / pers * 100:.0f} % lower than persistence.")
'''

JUDGES_REGION = '''
# @@perregion@@
rows = []
for region, d in test["per_region"].items():
    rows.append({"region": region, "model": d["model"]["mae_kw"], "persistence": d["persistence"]["mae_kw"],
                 "clear-sky persistence": d["clear_sky"]["mae_kw"], "model accuracy (%)": accuracy(d["model"]["mae_kw"])})
display(pd.DataFrame(rows).round(2).set_index("region"))
'''

JUDGES_CURVE = '''
# @@curve@@
curve = pd.DataFrame(meta["learning_curve"])
curve["accuracy (% of 50 kW)"] = accuracy(curve["model_mae_kw"])
display(curve[["days", "train_rows", "model_mae_wm2", "model_mae_kw", "accuracy (% of 50 kW)",
               "persistence_mae_wm2", "clear_sky_mae_wm2"]].round(2).set_index("days"))
'''

JUDGES_SAMPLE = '''
# @@sample@@
s = meta["report"]["sample"]
t = pd.to_datetime(s["times"]).tz_convert(config.LOCAL_TZ)
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(t, s["actual_kw"], color="black", lw=2, label="actual (derived from irradiance)")
ax.plot(t, s["predicted_kw"], color="tab:orange", lw=1.8, label="SolarCast model")
ax.plot(t, s["persistence_kw"], color="tab:blue", lw=1, ls="--", label="persistence")
ax.set_ylabel("PV power of a 50 kW system (kW)")
ax.set_title(f"One sample week in {s['region']}: next-hour forecast (NZ local time)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
'''

JUDGES_SERVE = '''
# @@serve@@
from solarcast.nz import webapp
url = webapp.serve(svc, port=8000, tunnel=True)

# @@big@@
display(HTML(f"""
<div style="padding:28px;margin:16px 0;border:4px solid #f59e0b;border-radius:16px;background:#fffbeb;text-align:center">
  <a href="{url}" target="_blank" style="font-size:36px;font-weight:700;color:#b45309">Open the SolarCast NZ dashboard</a>
  <div style="font-size:16px;margin-top:10px;color:#444">{url}</div>
</div>"""))
print(url)
'''

TRAIN_SETUP = f'''
import os, subprocess, sys

REPO = "/content/SolarCast-Lite"
URL = "{REPO_URL}.git"
# @@clone@@
if os.path.isdir(f"{{REPO}}/.git"):
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
elif not os.path.isdir(f"{{REPO}}/src"):
    subprocess.run(["git", "clone", URL, REPO], check=True)
# @@path@@
sys.path[:0] = [f"{{REPO}}/src", f"{{REPO}}/scripts"]
os.chdir(REPO)
'''

TRAIN_CONFIG = '''
# @@quick@@
QUICK = False

import json
from datetime import date, timedelta
import numpy as np
import pandas as pd
from solarcast.nz import config, data, model as nzmodel
from solarcast.nz.features import build_features
import train_nz            # the same helpers (split, GRID) as scripts/train_nz.py

end = date.today() - timedelta(days=6)          # the archive lags about 5 days behind real time
if QUICK:
    # @@quick_cfg@@
    regions, start = ["Auckland", "Christchurch"], end - timedelta(days=119)
    a, b = start + timedelta(days=72), start + timedelta(days=96)
    config.SPLITS.update(train=(str(start), str(a)), val=(str(a), str(b)), test=(str(b), str(end + timedelta(days=1))))
    GRID = train_nz.GRID[:2]
    model_path, meta_path = "/content/quick_model.joblib", "/content/quick_model.json"
else:
    # @@full_cfg@@
    regions, start = list(config.REGIONS), config.HISTORY_START
    GRID = train_nz.GRID
    model_path = meta_path = None
from pathlib import Path
model_path = Path(model_path) if model_path else None
meta_path = Path(meta_path) if meta_path else None
print("QUICK" if QUICK else "FULL", "| regions:", len(regions), "|", start, "->", end, "| configs to try:", len(GRID))
'''

TRAIN_FETCH = '''
# @@fetch@@
long_df = data.fetch_history(str(start), str(end), regions)
print(long_df.shape, long_df["time_utc"].min(), "->", long_df["time_utc"].max())
long_df.head()
'''

TRAIN_FEAT = '''
# @@feat@@
feat = build_features(long_df)
train, val, test = (train_nz.split(feat, k) for k in ("train", "val", "test"))
print("rows  train / val / test:", len(train), len(val), len(test), "| features:", len(nzmodel.FEATURES))

# @@baseline@@
base = {"persistence": nzmodel.persistence(val), "clear_sky": nzmodel.clear_sky_persistence(val)}
print("baselines on VAL:", {k: round(float(abs(val["target_ghi"] - p).mean()), 1) for k, p in base.items()}, "(MAE W/m2, all hours)")
'''

TRAIN_TUNE = '''
# @@tune@@
best, tuning = None, []
for params in GRID:
    m = nzmodel.train(train, **params)
    ev = nzmodel.evaluate(val, {"model": nzmodel.predict(m, val), **base})
    mae = ev["overall"]["model"]["mae_wm2"]
    tuning.append({**params, "val_mae_wm2": mae})
    print(f"{params} -> val MAE {mae:.2f} W/m2")
    if best is None or mae < best[0]:
        best = (mae, params, m, ev)
_, params, model, val_ev = best
train_nz.show("VALIDATION (chosen config)", val_ev)
o = val_ev["overall"]
beats = o["model"]["mae_wm2"] < min(o["persistence"]["mae_wm2"], o["clear_sky"]["mae_wm2"])
assert beats or QUICK, "the model must beat both baselines on the validation period"
print("chosen:", params, "| beats both baselines on VAL:", beats)
'''

TRAIN_TEST = '''
# @@test@@
test_ev = nzmodel.evaluate(test, nzmodel.all_predictions(model, test))
train_nz.show("TEST (reported once)", test_ev)
m = test_ev["overall"]["model"]
print(f"Test MAE {m['mae_kw']:.2f} kW = {100 - m['mae_kw'] / config.SYSTEM_KW * 100:.1f} % accuracy of the 50 kW system")
'''

TRAIN_CURVE = '''
# @@lc_old@@
old_meta = nzmodel.load_meta()
curve = old_meta.get("learning_curve")
if curve:
    lc = pd.DataFrame(curve)
    lc["accuracy (% of 50 kW)"] = 100 - lc["model_mae_kw"] / config.SYSTEM_KW * 100
    display(lc[["days", "train_rows", "model_mae_wm2", "model_mae_kw", "accuracy (% of 50 kW)"]].round(2).set_index("days"))
else:
    print("no learning curve stored in the JSON")
'''

TRAIN_SAVE = '''
# @@save@@
report = nzmodel.build_report(model, val, test)
nzmodel.save(model, {"val": val_ev, "test": test_ev}, model_path=model_path, meta_path=meta_path,
             extra={"chosen_params": params, "tuning": tuning, "data_end": str(end), "report": report,
                    "learning_curve": curve, "data_source": "Open-Meteo (ERA5 reanalysis), CC BY 4.0"})
print("saved to", model_path or config.MODEL_PATH, "and", meta_path or config.MODEL_META)
'''

TRAIN_BACKUP = '''
# @@backup@@
BACKUP = False
if BACKUP and not QUICK:
    import subprocess
    from google.colab import userdata
    # @@token@@
    token = userdata.get("GITHUB_TOKEN")

    def git(*args):
        r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(token, "***").strip())

    git("config", "user.name", "SolarCast Colab")
    git("config", "user.email", "solarcast-colab@users.noreply.github.com")
    git("add", "models/nz_model.joblib", "models/nz_model.json")
    git("commit", "-m", "model(nz): retrained on Colab")
    git("push", f"https://x-access-token:{token}@github.com/nashwu0117/SolarCast-Lite.git", "HEAD:main")
else:
    print("backup skipped (BACKUP is False or QUICK is True)")
'''

# ---------------------------------------------------------------------------------------------------------------
# Markdown (per language)
# ---------------------------------------------------------------------------------------------------------------
BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def colab_url(name: str) -> str:
    return f"https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/{name}"


MD = {
 "judges_title": {
  "en": """# SolarCast NZ: when should a small workshop charge its EV?

**Veritas AI High School Hackathon 2026, AI + Climate.** {badge}

**The story.** A small workshop in New Zealand has a **50 kW rooftop solar system** and an **electric vehicle** that needs
about 40 kWh a day. Solar power changes hour by hour with the clouds. If the EV charges while the sun is strong, the
workshop uses its own clean, free power; if it charges when the sun is weak, it pulls power from the grid (it costs money
and carries CO2). So the question is: **how much solar power will we have in the next hour, and should we charge now?**

**What SolarCast NZ does.** A machine-learning model (gradient boosting, scikit-learn) forecasts the **next hour** of solar
power for 14 New Zealand regions, and the dashboard turns that forecast into simple EV charging advice.

**How to run it:** `Runtime > Run all`. It takes about 2-3 minutes. The **last cell** starts the dashboard and prints a big link.""",
  "zh-TW": """# SolarCast NZ：小型工坊什麼時候該幫電動車充電？

**Veritas AI 高中黑客松 2026，主題 AI + 氣候。** {badge}

**情境故事。** 紐西蘭有一間小型工坊，屋頂裝了 **50 kW 的太陽能系統**，還有一輛每天大約需要 40 kWh 的**電動車**。
太陽能發電量會隨著雲層一小時一小時地變化。如果電動車在太陽很大的時候充電，工坊就能用自己乾淨又免費的電；
如果在太陽很弱時充電，就得向電網買電（要花錢，也會產生 CO2）。所以問題是：**下一個小時會有多少太陽能？現在該不該充電？**

**SolarCast NZ 做了什麼。** 我們用機器學習模型（梯度提升，scikit-learn）預測紐西蘭 14 個區域**下一小時**的太陽能發電量，
再由儀表板把預測轉成簡單的電動車充電建議。

**怎麼執行：** 選 `執行階段 > 全部執行`，大約 2 到 3 分鐘。**最後一個儲存格**會啟動儀表板，並印出很大的連結。"""},
 "setup": {"en": "## 1. Setup\nClone the public repository (shallow), install Flask, and use the same scikit-learn version the model was trained with.",
           "zh-TW": "## 1. 環境設定\n以淺層 clone 取得公開儲存庫、安裝 Flask，並確保 scikit-learn 版本與模型訓練時相同。"},
 "load": {"en": "## 2. Load the model and the replay data\nThe repository ships a small replay file (the test period only), so you do **not** wait for a long weather download.",
          "zh-TW": "## 2. 載入模型與回放資料\n儲存庫內附一個很小的回放檔（只有測試期間），所以**不需要**等很久的天氣資料下載。"},
 "report": {"en": """## 3. Model report (test period, never used for training or tuning)
- **Persistence**: "next hour = this hour". **Clear-sky persistence**: keep the current clear-sky index. Both are baselines the model must beat.
- **MAE** is the average error in kW (lower is better). **Accuracy as a share of the 50 kW system = 100 - nMAE**, with nMAE = MAE / 50 kW.""",
            "zh-TW": """## 3. 模型報告（測試期間，從未用於訓練或調參）
- **持續性預測（Persistence）**：「下一小時 = 這一小時」。**晴空持續性預測**：保留目前的晴空指數。這兩個是模型必須打敗的基準線。
- **MAE** 是平均誤差（單位 kW，越小越好）。**以 50 kW 系統為基準的準確度 = 100 - nMAE**，其中 nMAE = MAE / 50 kW。"""},
 "overall": {"en": "### Overall test results", "zh-TW": "### 整體測試結果"},
 "region": {"en": "### Per region", "zh-TW": "### 各區域結果"},
 "curve": {"en": "### Learning curve\nMore training days means a lower error, and the gain flattens after about a year of data.",
           "zh-TW": "### 學習曲線\n訓練天數越多，誤差越低；大約一年的資料之後，進步就趨於平緩。"},
 "sample": {"en": "### One sample week", "zh-TW": "### 取樣一週"},
 "notes": {
  "en": """## 4. Honest notes
- The weather is **Open-Meteo modelled data (ERA5-based reanalysis)**, not measurements from real solar plants.
- The PV power is **derived from irradiance** with a stated formula (50 kW x GHI/1000 x 0.80 performance ratio, with a small temperature loss). New Zealand has no public per-site hourly PV output data.
- A real plant would be harder: shading, dirt, inverter limits, outages and forecast errors all add noise.
- This is a **prototype**. It is **not energy or financial advice**.""",
  "zh-TW": """## 4. 誠實說明
- 天氣資料是 **Open-Meteo 的模型資料（以 ERA5 為基礎的再分析資料）**，不是真實太陽能電廠的量測值。
- 發電量是用**日射量換算**出來的（50 kW × GHI/1000 × 0.80 系統效率，再扣掉一點高溫損失）。紐西蘭沒有公開的單一電站每小時發電資料。
- 真實的電廠會更難預測：遮蔭、灰塵、變流器上限、停機與預報誤差都會增加雜訊。
- 這只是**原型**，**不是能源或財務建議**。"""},
 "credits": {
  "en": """## Credits
Weather data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0) | Machine learning: scikit-learn | Web server: Flask |
Map: Leaflet with OpenStreetMap | Charts: Chart.js | Public link: Cloudflare cloudflared |
AI tool: Claude Code (Anthropic) helped write and test the code.""",
  "zh-TW": """## 致謝
天氣資料：[Open-Meteo](https://open-meteo.com)（CC BY 4.0）｜機器學習：scikit-learn｜網頁伺服器：Flask｜
地圖：Leaflet 與 OpenStreetMap｜圖表：Chart.js｜公開連結：Cloudflare cloudflared｜
AI 工具：Claude Code（Anthropic）協助撰寫與測試程式。"""},
 "final": {
  "en": """## 5. FINAL STEP: start the dashboard
Run the cell below. It prints a **big link** to the SolarCast NZ dashboard (map of New Zealand: hover a region for weather and
generation, click it for the day chart, EV advice and savings; the Model tab shows the report above; use the Replay slider
or the Live button). Keep this notebook running while you use the link.""",
  "zh-TW": """## 5. 最後一步：啟動儀表板
執行下面的儲存格，會印出一個**很大的連結**，連到 SolarCast NZ 儀表板（紐西蘭地圖：滑鼠移到區域上看天氣與發電量，
點擊區域看當天圖表、電動車建議與節省金額；Model 分頁有上面的模型報告；可以用 Replay 滑桿或 Live 按鈕）。
使用連結的時候，請讓這個筆記本保持在執行狀態。"""},
 # ----- train -----
 "train_title": {
  "en": """# SolarCast NZ: training notebook

Fetch Open-Meteo history for 14 New Zealand regions, build features, compare with two baselines, **tune on the validation
period only**, look at the **test period once**, and save the model. {badge}

Set `QUICK = True` in the config cell for a fast smoke run (2 regions x 120 days, about a minute, and the repository model is not overwritten).
A full run downloads 5 years of history (it can take about 30 minutes because of Open-Meteo rate limits); judges do not need it: use the Judges notebook.""",
  "zh-TW": """# SolarCast NZ：訓練筆記本

下載紐西蘭 14 個區域的 Open-Meteo 歷史資料、建立特徵、和兩個基準線比較、**只用驗證集調參數**、
**只看一次測試集**，最後儲存模型。{badge}

在設定儲存格把 `QUICK = True`，就能快速跑一次冒煙測試（2 個區域 × 120 天，約一分鐘，而且不會覆蓋儲存庫裡的模型）。
完整執行要下載 5 年的歷史資料（因為 Open-Meteo 有流量限制，可能要花約 30 分鐘）；評審不需要這個，請直接用 Judges 筆記本。"""},
 "t_setup": {"en": "## 1. Setup", "zh-TW": "## 1. 環境設定"},
 "t_cfg": {"en": "## 2. Config\n`QUICK` is `False` by default (full run).", "zh-TW": "## 2. 設定\n`QUICK` 預設是 `False`（完整執行）。"},
 "t_fetch": {"en": "## 3. Fetch history\nOpen-Meteo radiation at time T is the mean over the **preceding** hour; the code respects that.",
             "zh-TW": "## 3. 下載歷史資料\nOpen-Meteo 在時間 T 的日射量，是「前一個小時」的平均值；程式碼有遵守這一點。"},
 "t_feat": {"en": "## 4. Features and baselines\nThe target is the mean irradiance of the **next** hour.",
            "zh-TW": "## 4. 特徵與基準線\n預測目標是**下一個小時**的平均日射量。"},
 "t_tune": {"en": "## 5. Train and tune (validation only)", "zh-TW": "## 5. 訓練與調參（只用驗證集）"},
 "t_test": {"en": "## 6. Test once", "zh-TW": "## 6. 只測一次"},
 "t_curve": {"en": "## 7. Learning curve summary\nRead from the JSON of the committed model.", "zh-TW": "## 7. 學習曲線摘要\n從已提交模型的 JSON 讀取。"},
 "t_save": {"en": "## 8. Save the model", "zh-TW": "## 8. 儲存模型"},
 "t_backup": {
  "en": """## 9. Back up the model to GitHub (optional)
Colab VMs are temporary, so push the new model to GitHub. Add a Colab **Secret** named `GITHUB_TOKEN` (key icon on the left; a GitHub token with
write access to the repository) and enable notebook access. The token is never printed. Then set `BACKUP = True`.""",
  "zh-TW": """## 9. 把模型備份到 GitHub（選用）
Colab 的虛擬機是暫時的，所以要把新模型 push 到 GitHub。請在 Colab 左側的鑰匙圖示新增一個名為 `GITHUB_TOKEN` 的 **Secret**
（對該儲存庫有寫入權限的 GitHub token），並開啟筆記本存取權限。token 不會被印出來。然後把 `BACKUP` 改成 `True`。"""},
}


def nb(cells: list, lang: str) -> dict:
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        out.append(cell)
    return {"cells": out, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"provenance": []}}}


def md(key: str, lang: str, **kw) -> tuple:
    return ("markdown", MD[key][lang].format(**kw) if kw else MD[key][lang])


def build(lang: str) -> dict:
    suffix = "" if lang == "en" else ".zh-TW"
    C = lambda s: ("code", code(s, lang))
    judges = [
        md("judges_title", lang, badge=BADGE.format(url=colab_url(f"SolarCast_NZ_Judges{suffix}.ipynb"))),
        md("setup", lang), C(JUDGES_SETUP),
        md("load", lang), C(JUDGES_LOAD),
        md("report", lang), md("overall", lang), C(JUDGES_OVERALL),
        md("region", lang), C(JUDGES_REGION),
        md("curve", lang), C(JUDGES_CURVE),
        md("sample", lang), C(JUDGES_SAMPLE),
        md("notes", lang), md("credits", lang),
        md("final", lang), C(JUDGES_SERVE),
    ]
    train = [
        md("train_title", lang, badge=BADGE.format(url=colab_url(f"train{suffix}.ipynb"))),
        md("t_setup", lang), C(TRAIN_SETUP),
        md("t_cfg", lang), C(TRAIN_CONFIG),
        md("t_fetch", lang), C(TRAIN_FETCH),
        md("t_feat", lang), C(TRAIN_FEAT),
        md("t_tune", lang), C(TRAIN_TUNE),
        md("t_test", lang), C(TRAIN_TEST),
        md("t_curve", lang), C(TRAIN_CURVE),
        md("t_save", lang), C(TRAIN_SAVE),
        md("t_backup", lang), C(TRAIN_BACKUP),
        md("credits", lang),
    ]
    return {f"SolarCast_NZ_Judges{suffix}.ipynb": nb(judges, lang), f"train{suffix}.ipynb": nb(train, lang)}


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    for lang in LANGS:
        for name, notebook in build(lang).items():
            (NB_DIR / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
            print("wrote", NB_DIR / name)
