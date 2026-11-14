# 團隊 Colab 操作說明

[English](COLAB_GUIDE.md) | **繁體中文**

所有人都在 Google Colab 上工作，不需要在自己電腦上跑任何東西。程式碼放在**公開**的 GitHub repo
`nashwu0117/SolarCast-Lite`。請先完整讀一次，之後隨時開著參考。

**截止時間：2026 年 11 月 15 日（日）晚上 11:59 ET（= 紐西蘭時間 11 月 16 日（一）下午 5:59）。**
請提早幾小時先交第一版，之後有改進再重新提交。

## 1. 兩個 notebook

| Notebook | 用途 | English | 繁體中文 |
|---|---|---|---|
| **評審用**（交件） | 從 GitHub 載入訓練好的模型、顯示測試結果、開啟有 AI 聊天的地圖儀表板 | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) |
| **訓練用** | 下載 Open-Meteo 歷史資料、訓練並評估模型、把模型備份到 GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) |

**交件用的是英文版評審 notebook。** 中文版是同一份程式碼加上中文說明，拿來測試和學習。

## 2. 執行評審 notebook

1. 打開連結。它會向 Colab 要求免費的 **T4 GPU**（「執行階段 > 變更執行階段類型」會顯示 T4 GPU）。如果 Colab 說
   沒有 GPU 可用，就選**不使用 GPU 連線**：所有功能照常運作，只是 AI 聊天比較慢。
2. 「執行階段 > **全部執行**」。
3. 大約 2–3 分鐘後，第 5 節會印出很大的**儀表板連結**（免費的 Cloudflare tunnel），點開它。
4. 儀表板右下角的 **Ask SolarCast AI** 大約再過幾分鐘、模型下載並載入完成後，就會變成真正的 AI 聊天；
   在那之前會用固定範本回答。第 6 節會等它準備好，並問兩個測試問題（一個正常、一個它必須拒絕）。
5. 使用連結的時候，請保持 notebook 分頁開著：儀表板是跑在那台 Colab 機器上。

可以在儀表板上試試：把 **Replay** 滑桿拉到紐西蘭白天的時段（Live 模式在晚上每個地區都是 0 kW，這是正確的）、
滑鼠移到地區上、點地區看詳細資料、打開 **Model** 分頁，然後問聊天「下一小時哪個地區發電最多？」或「基督城今天可以省多少錢？」。

## 3. 執行訓練 notebook

- `QUICK = True`（設定儲存格）：1 分鐘的冒煙測試，2 個地區 × 120 天，不會動到 repo 裡的模型。
- `QUICK = False`：完整執行，14 個地區、從 2022 年開始。因為 Open-Meteo 有流量限制，下載可能要約 30 分鐘。
- 備份到 GitHub（最後一格，`BACKUP = True`）需要 token，見第 6 節。只有在**驗證期**比目前模型好超過 1 W/m² 的
  新模型才能換上；每個換上的模型，測試期只看一次。

## 4. 修改程式碼

- Python 程式碼在 `src/solarcast/nz/`（config、data、features、ensemble、model、service、webapp、app、assistant、
  llm_setup）。打開左側檔案瀏覽器（資料夾圖示），進入 `/content/SolarCast-Lite/src/solarcast/nz`，雙擊檔案編輯，
  用 Ctrl/Cmd+S 存檔，然後「執行階段 > **重新啟動工作階段**」再重跑（Colab 的 Python 3.13 不能用 autoreload）。
- **notebook 是自動產生的。** 不要手動改 `.ipynb`：請改 `scripts/make_nz_notebooks.py`（程式碼儲存格是共用的，
  中英文說明都寫在裡面），再執行 `python scripts/make_nz_notebooks.py`。這樣兩種語言永遠一致。
- 各模組的介面寫在 `docs/NZ_CONTRACT.md`。要改介面時，先改這份文件並通知大家。

## 5. 在 Colab 跑測試

```python
!pip -q install pytest flask
!cd /content/SolarCast-Lite && PYTHONPATH=src python -m pytest -q tests/nz
```

57 個測試應該在 20 秒左右全部通過。測試完全離線：不會下載模型，也不用連網。

## 6. 把修改推上 GitHub

1. 建立 token（只需做一次）：GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → Generate new token。Repository access 只選 `SolarCast-Lite`，Permissions 選
   **Contents: Read and write**。
2. 在 Colab 左側點鑰匙圖示（**Secrets**）→ **Add new secret**：名稱 `GITHUB_TOKEN`，值填你的 token，並打開
   **Notebook access**。絕對不要把 token 貼進儲存格、commit 或聊天裡。
3. 把下面這段貼到新的儲存格（它會從 Secrets 讀 token，而且絕不會印出來）：

```python
import subprocess
from google.colab import userdata

NAME = "Your Name"
EMAIL = "you@users.noreply.github.com"
MESSAGE = "fix(nz): describe your change"

REPO = "/content/SolarCast-Lite"
token = userdata.get("GITHUB_TOKEN")
url = f"https://x-access-token:{token}@github.com/nashwu0117/SolarCast-Lite.git"

def git(*args):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***").strip())
    return r.returncode

git("config", "user.name", NAME)
git("config", "user.email", EMAIL)
git("add", "src", "tests", "docs", "scripts", "notebooks", "README.md", "README.zh-TW.md")
git("commit", "-m", MESSAGE)
git("pull", "--rebase", url, "main")
git("push", url, "HEAD:main")
```

commit 訊息用英文，照 Conventional Commits 格式寫（`feat(nz): ...`、`fix(nz): ...`、`docs: ...`）。如果
`pull --rebase` 出現衝突，先停下來，聯絡那個檔案的負責人。

## 7. 檔案分工

**同一個檔案同一時間只能一個人改。** 要改別人負責的檔案前，先跟對方說。

| 範圍 | 檔案 |
|---|---|
| 資料與特徵 | `src/solarcast/nz/data.py`、`features.py`、`data/nz/`、`scripts/build_replay_data.py` |
| 模型與評估 | `src/solarcast/nz/model.py`、`ensemble.py`、`scripts/train_nz.py`、`docs/MODEL*.md` |
| 儀表板與 AI 聊天 | `src/solarcast/nz/service.py`、`webapp.py`、`assistant.py`、`llm_setup.py`、`app.py` |
| notebook 與報告 | `scripts/make_nz_notebooks.py`、`README*.md`、`NOTICE*.md`、一頁 write-up 與影片 |

共用檔案，大家同意才能改：`src/solarcast/nz/config.py` 和 `docs/NZ_CONTRACT.md`。

黃金規則：**絕對不要用測試期調參數。** 用 2022–2024 年訓練、用 2025 年 1–6 月做選擇、2025 年 7 月–2026 年 10 月只報告一次。

## 8. 交件前

1. 用無痕視窗打開**英文版評審** notebook 連結（不登入也要能打開，所以 repo 必須保持公開），然後「執行階段 >
   **全部執行**」。必須全部跑完沒有錯誤、儀表板連結打得開、聊天能回答。
2. repo 裡的 notebook 是**含輸出**存檔的（briefs 要求）。如果有修改，要在 Colab 上重跑，再連同輸出一起存檔。
3. 檢查 README 的結果和來源清單（`NOTICE.md`：所有資料集、預訓練模型和 AI 工具，附連結）。
4. 繳交表單：程式碼連結（英文版評審 notebook 或 repo）、write-up 連結（README 或設為「知道連結的人都能檢視」的
   Google Doc）、選填影片（最多 2 分鐘，YouTube 不公開或 Drive）、賽道 **AI + Climate**、每位隊員的姓名、email
   和學校，以及我們加了什麼（「Make it your own」：紐西蘭、下一小時而不是隔天、電動車充電決策、AI 聊天）。
5. 在**紐西蘭時間 11 月 16 日（一）下午 5:59** 前交件，最好提早幾小時。
