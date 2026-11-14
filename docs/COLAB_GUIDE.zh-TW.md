# 團隊 Colab 操作說明

[English](COLAB_GUIDE.md) | **繁體中文**

所有人都在 Google Colab 上工作。程式碼放在 GitHub repo `nashwu0117/SolarCast-Lite`（目前是**私有**）。
請先完整讀一次，之後隨時開著參考。

**截止時間：2026 年 11 月 15 日（日）晚上 11:59 ET（= 台灣時間 11 月 16 日（一）中午 12:59）。**

## 1. 取得私有 repo 的權限（只需做一次）

1. 把你的 GitHub 帳號給 repo 擁有者，並接受 **collaborator 邀請**（請查收 email）。
2. 打開下面任一個 notebook 連結。Colab 會要求連結 GitHub：按 **Authorize**（允許存取私有 repo）。
3. 建立一組 token，讓 notebook 可以 clone 和 push：
   - GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token。
   - Repository access：選 *Only select repositories*，勾選 `SolarCast-Lite`。
   - Permissions：**Contents: Read and write**。
   - 複製 token（只會顯示一次）。
4. 在 Colab 左側欄點鑰匙圖示（**Secrets**）→ **Add new secret**：
   名稱 `GITHUB_TOKEN`，值填你的 token，並打開 **Notebook access**。
   絕對不要把 token 貼進 notebook 的程式格或聊天裡。

## 2. 開啟 notebook

直接點連結即可：

| Notebook | 連結 |
|---|---|
| main（交件用） | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/main.ipynb |
| 01 資料探索 | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/01_data_exploration.ipynb |
| 02 建模 | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/02_modeling.ipynb |
| 03 Demo | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/03_demo.ipynb |

或者：Colab → File → Open notebook → GitHub 分頁 → 選 `nashwu0117/SolarCast-Lite`。

先執行第一格程式（setup）。它會把 repo clone 到 `/content/SolarCast-Lite`（已存在就 pull 最新版），
並把 `src/` 加進 Python 路徑。接著由上往下執行其餘程式格。

## 3. 把 notebook 的修改存回 GitHub

File → **Save a copy in GitHub**。選 repo `nashwu0117/SolarCast-Lite`、branch **main**、相同的檔案路徑，
寫一句簡短的 commit 訊息（例如 `02: try deeper trees on val`），可視需要勾選附上 Colab 連結。
別人負責的 notebook，請先問過對方再存。

## 4. 修改 Python 檔（`src/solarcast/*.py`）

1. 打開左側的檔案瀏覽器（資料夾圖示），進入 `/content/SolarCast-Lite/src/solarcast`。
2. 雙擊檔案即可在 Colab 中編輯，用 Ctrl/Cmd+S 存檔。
3. setup cell 會嘗試開啟 autoreload，讓下一格直接用到新程式碼。目前 Colab 的 IPython 在 Python 3.12 以上無法載入
   autoreload，這時 setup cell 會印出「autoreload unavailable」。遇到這種情況，每次修改後請 Runtime →
   **Restart session**，再重新執行 setup cell。

### 把修改推上 GitHub

把下面這段貼到 notebook 最下方的**新程式格**（token 是從 Secrets 讀取，不要把填好 token 的版本存進 notebook）。
填上你的名字、email 和 commit 訊息：

```python
import subprocess
from google.colab import userdata

NAME = "Your Name"
EMAIL = "you@example.com"
MESSAGE = "describe your change"

REPO = "/content/SolarCast-Lite"
token = userdata.get("GITHUB_TOKEN")
url = f"https://x-access-token:{token}@github.com/nashwu0117/SolarCast-Lite.git"

def git(*args, hide=True):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(token, "***")
    print(out.strip())
    return r.returncode

git("config", "user.name", NAME)
git("config", "user.email", EMAIL)
git("add", "src", "tests", "docs", "README.md", "README.zh-TW.md", "notebooks", "scripts")
git("commit", "-m", MESSAGE)
git("pull", "--rebase", url, "main")
git("push", url, "HEAD:main")
```

如果 `pull --rebase` 出現衝突，先停下來，聯絡那個檔案的負責人。token 不會被印出來。

## 5. 取得隊友的修改

重新執行 setup cell（它會跑 `git pull --ff-only`），如果程式碼看起來是舊的，再 Runtime → **Restart session**。
如果 pull 顯示無法 fast-forward，代表你有本地修改：先推上去（第 4 節），再 pull。

## 6. 檔案負責規則

**同一個檔案同一時間只能一個人改。** 要改別人負責的檔案前，先跟對方說。

| 角色 | 負責檔案 |
|---|---|
| Data Lead | `src/solarcast/data.py`、`features.py`、`baselines.py`、`notebooks/01_data_exploration.ipynb` |
| ML Lead | `src/solarcast/models.py`、`evaluate.py`、`pipeline.py`、`notebooks/02_modeling.ipynb` |
| Product & Demo Lead | `src/solarcast/demo.py`、`notebooks/03_demo.ipynb`、`notebooks/main.ipynb`、`README.md` |

共用檔案、需要大家同意才能改：`src/solarcast/config.py` 和 `docs/CONTRACT.md`。如果介面要改，
先改 `docs/CONTRACT.md` 並通知所有人。

黃金規則：**絕對不要用測試集調參數。** 用 `train` 訓練、用 `val` 做選擇、`test` 只在最後報告一次。

## 7. 在 Colab 跑測試

```python
!pip -q install pytest
!cd /content/SolarCast-Lite && PYTHONPATH=src python -m pytest -q
```

因為套件是直接從 `src/` 使用、沒有 pip 安裝，所以需要 `PYTHONPATH=src`。30 個測試應該在 10 秒左右全部通過。

## 8. 交件前

1. 在 `main.ipynb`：Runtime → **Restart session and run all**，必須全部跑完、沒有錯誤。
2. 確認 Demo 能啟動並印出公開連結。
3. 把含輸出的 notebook 存回 GitHub（第 3 節），並確認 README 的結果區塊已填入真實數字。
4. 在 **11 月 15 日（日）晚上 11:59 ET（台灣 11 月 16 日（一）中午 12:59）** 前交件，最好提早幾小時完成。
