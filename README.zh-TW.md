# SolarCast NZ

[English](README.md) | **繁體中文**

你的太陽能板下一小時能發多少電？SolarCast NZ 用機器學習預測紐西蘭任何地點的答案，並在互動地圖上依你的系統大小顯示
（例如家用 5 kW 或商用 50 kW），另外用電動車智慧充電當實際應用範例。
Veritas AI 國際高中黑客松 2026，**AI + Climate** 賽道。

**網站：** [https://nashwu0117.github.io/SolarCast-Lite/](https://nashwu0117.github.io/SolarCast-Lite/) &middot; **授權：** MIT（開源）

**評審請從這裡開始** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) 或 [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges_Kaggle.zh-TW.ipynb)
（兩個選你喜歡的、或你有 GPU 額度或付費方案的那一個就好：跑的是同一份 notebook，打開的儀表板也一樣。
Kaggle：在 **Settings** 選 **Accelerator: GPU T4 x2**、**Internet: On**（開網路需要先驗證手機號碼），再按 **Run All**。）
Colab：Runtime > **Run all**（notebook 會要求免費的 T4 GPU，只給 AI 聊天用）。大約三分鐘內，notebook 會從本 repo 載入
我們訓練好的模型、顯示它的準確度，並印出儀表板的公開連結（免費的 Cloudflare tunnel）；AI 聊天再過幾分鐘就會準備好。

| Notebook | 用途 | Colab（English） | Colab（繁體中文） | Kaggle（English） | Kaggle（繁體中文） |
|---|---|---|---|---|---|
| 評審用 | 載入訓練好的模型、顯示結果、開啟儀表板（GPU T4） | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) | [Open](https://kaggle.com/kernels/welcome?src=https://github.com/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges_Kaggle.ipynb) | [開啟](https://kaggle.com/kernels/welcome?src=https://github.com/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges_Kaggle.zh-TW.ipynb) |
| 訓練用 | 抓資料、訓練並評估模型、把模型備份到 GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) | [Open](https://kaggle.com/kernels/welcome?src=https://github.com/nashwu0117/SolarCast-Lite/blob/main/notebooks/train_Kaggle.ipynb) | [開啟](https://kaggle.com/kernels/welcome?src=https://github.com/nashwu0117/SolarCast-Lite/blob/main/notebooks/train_Kaggle.zh-TW.ipynb) |

## 使用者與決策
紐西蘭任何裝了屋頂太陽能的人，從裝 5 kW 的家庭到裝 50 kW 的學校或店家，如果能在太陽真的出來的時段使用大電器
（電動車充電、熱水器、熱泵、洗碗機、工作機台），就能把自家的太陽能用得最充分。但雲很難猜。SolarCast 幫全國任何地點
回答一個問題：**我的太陽能板下一小時能發多少 kW？**

**一句話介紹：** *SolarCast 告訴紐西蘭任何人，他們的太陽能板下一小時能發多少電，讓他們用自家的陽光，而不是向電網買電。*

**實際應用範例（有實測）：** 一間屋頂有 50 kW 太陽能板的加工店，電動車每個工作天大約要充 40 kWh。
讓預測決定什麼時候充電，可以減少電動車向電網買的電（見下方「預測真的讓決策更好嗎？」）。

## 儀表板
- **你的系統大小**（3、5、10、50 或 100 kW）：頁面上所有 kW 和度數都換算成你的系統。發電量和太陽能板大小成正比；
  模型預測的是日照量，再依你選的大小換算成電力。
- **紐西蘭地圖**：14 個地區，依「你的太陽能板下一小時能發多少電」上色。
- **全國格子**：同一個模型跑涵蓋全國的 137 個陸地格子（0.5 度，約 50 公里），用熱力圖顯示。網頁打開前就算好，
  之後每小時 :50 分重算，讓下一個小時開始前就準備好；即時模式會自己更新。
- **AI 全國摘要**（地圖上方）：本機 AI 模型只根據格子數字寫兩三句話（哪裡最晴朗、北島和南島比較、一套家用 5 kW 系統能發多少）；
  模型準備好之前先顯示固定範本。
- **滑鼠移到地區上**：顯示當下天氣，以及有沒有在發電、發多少。
- **點紐西蘭任何地方**（不只 14 個圓點）：會放一個藍色圖釘，下載那個地點的 Open-Meteo 天氣並用同一個模型預測
  （模型借用最近地區的氣候代號；準確度顯示的是那個地區的測試結果）。
- **點擊地區**：顯示你的太陽能板下一小時的發電量、當天預測總度數和以電價計算的價值、當天實際與預測的發電曲線，
  以及應用範例（下一小時的電動車充電建議與原因、當天省下的電、錢和 CO₂）。網頁一打開就會選好目前最晴朗的地區，
  不用點也看得到；還有一張動畫，畫出電從太陽流到你的太陽能板、家裡，以及（值得充電時）電動車。
- **用你的太陽能板幫電動車充電**：選特斯拉 Model Y 或比亞迪 Atto 3（2025 年紐西蘭最暢銷的兩款電動車）、
  現在的電量、要充到多少，以及充電器。依每小時的預測算出太陽能板什麼時候能充完（或能充到哪裡、還差多少）、
  增加的續航里程、省下的錢，以及同樣的電改用電網要充多久。電池會有充電動畫，長條圖顯示每小時的太陽能用在哪裡
  （家裡、車子、剩下）。晚上則改顯示「早上一出太陽就插上」的整天估算。AI 聊天也能回答同樣的問題。
- **回放**測試期間的任一個小時，或切換到**即時**（紐西蘭時間；晚上太陽能為 0）。
- **模型分頁**：我們模型和兩個基準方法的準確度比較、各地區表現、一週範例，以及模型最依賴哪些因素。
- **Ask SolarCast AI**（右下角）：用中文或英文解釋資料的聊天，會依你選的系統大小回答，例如「我的太陽能板下一小時在基督城能發多少電？」。
  它使用開源模型（Colab T4 GPU 上跑 Qwen3 14B，較小的 GPU 跑 8B，只有 CPU 時跑 1.7B），**只在 notebook 那台機器上本機執行**：
  不能上網、沒有工具、不需要 API 金鑰。伺服器先用和地圖相同的程式查出數字，模型只負責解釋；關鍵字過濾、Llama Guard 3
  安全分類器和系統規則會拒絕離題或有害的要求；提問也有頻率限制。預測本身是我們自己的模型，聊天只是架在上面的介面。

## 運作方式
- **資料**：Open-Meteo 提供的紐西蘭 14 個城市逐時天氣，2022 年到 2026 年 11 月（CC BY 4.0）。
- **任務**：在第 *t* 小時結束時，預測第 *t+1* 小時的日照量（GHI），再換算成一套 50 kW 參考系統的發電量（其他大小按比例換算；誤差也用「佔容量百分比」報告）。
  輸入只用第 *t* 小時以前已知的資料，加上太陽位置（有自動測試確保沒有偷看未來）。
- **基準方法**：Persistence（下一小時 = 這一小時）與晴空 Persistence（維持這一小時的雲量程度）。
- **模型（v4）**：scikit-learn 的梯度提升樹，所有地區共用一個模型，參數只用驗證期挑選。
  它把兩個模型取平均：一個直接預測日照量，一個預測「雲多不多」（晴空指數，因為太陽的軌跡可以精確算出）。
  兩個模型也都參考 Open-Meteo 針對被預測那一小時的**前一天預報**（約 24 小時前發布，所以沒有偷看未來；
  2024 年起才有存檔，更早的資料是缺值）。
- **依時間切分**：2022–2024 年訓練、2025 年 1–6 月驗證、2025 年 7 月–2026 年 10 月測試（只報告一次）。

## 結果（測試期、白天時段、全部 14 個地區）

| 方法 | 平均誤差（50 kW 系統） | 佔容量 | 佔白天平均發電量（14.3 kW） |
|---|---|---|---|
| Persistence | 4.03 kW | 8.1% | 28.3% |
| 直接用 Open-Meteo 前一天預報 | 1.88 kW | 3.8% | 13.2% |
| 晴空 Persistence | 1.66 kW | 3.3% | 11.7% |
| SolarCast v1（只用到目前為止的天氣） | 1.19 kW | 2.4% | |
| **SolarCast v4（目前版本）** | **1.12 kW** | **2.2%** | **7.9%** |

**準確度怎麼看。**「97.8% 準確度」是 100% 減去「誤差佔 50 kW 容量的比例」。如果改用白天平均發電量當分母，誤差是 7.9%。
最公平的一個數字：v4 的誤差比晴空 Persistence（最強的簡單基準）**低 32.6%**，比 Open-Meteo 自己的前一天預報低 40.3%，
比 Persistence 低 72.2%。多雲時誤差比較大：雲量 60% 以下 15.8 W/m2，60% 以上 41.1 W/m2。

SolarCast v4 在每一個地區都贏過兩個基準方法，也贏過 v1（14/14）。學習曲線（針對 v1 量測）顯示，模型大約需要半年的歷史資料
才能贏過晴空 Persistence，超過一年後進步就很小。完整數據：`models/nz_model.json`、`experiments/results/nz_learning_curve.md`。

**v4 是怎麼選出來的（只看驗證期，門檻事先訂好：要比 v1 好超過 1 W/m2）：**
v2 更多訓練地點：沒有進步；v3 加入前一天預報：小幅進步（0.71 W/m2），不夠；v4 預報輸入加上晴空指數模型再和原始模型取平均：
28.98 對 30.35 W/m2（進步 1.37），升級。細節：`experiments/results/nz_v2.json`、`nz_v3.json`、`nz_v4.json`。
測試期只看了一次，而且只針對 v4。

**v4 之後的嘗試（v5，只看驗證期，門檻一樣：要比 v4 的 28.98 好超過 1 W/m2）：**加入「前一天的預報現在錯了多少」
（剛結束這一小時的預報和實測日照差多少）、預報本身在目標小時前後的變化趨勢，再加上五組參數調整。最好的是 28.71 W/m2
（更大的樹），只進步 0.26，所以**保留 v4**。模型其實已經從「現在的日照」加上「目標小時的預報」學到大部分這些資訊。
v5 完全沒有讀測試期。細節：`scripts/train_nz_v5.py`、`experiments/results/nz_v5.json`。

**v6（GPU，只看驗證期，門檻一樣）：**同樣的雙模型設計，改用能在 GPU 上訓練的 XGBoost，在 Colab T4 上訓練（scikit-learn 的模型只能用 CPU）。
在 v4 和 v5 的輸入上試了 48 組樹深、學習率、樹的數量和葉節點大小（GPU 上每組 11–82 秒，CPU 版 v4 訓練一次要 97 秒），
再和 v4 混合。最好的 XGBoost 是 28.63 W/m2，最好的混合（0.7 XGBoost + 0.3 v4）是 28.56，只進步 0.42，所以**再次保留 v4**。
加上 v5 的結果，這表示剩下的誤差主要來自天氣資料本身，而不是模型。細節：`scripts/train_nz_v6.py`、`experiments/results/nz_v6.json`。

## 在你自己的電腦上執行
需要 Python 3.10 以上：
```bash
git clone https://github.com/nashwu0117/SolarCast-Lite.git
cd SolarCast-Lite
python -m venv .venv && source .venv/bin/activate      # Windows：.venv\Scripts\activate
pip install -e .
python -m solarcast.nz                                 # 打開 http://127.0.0.1:8000
```
選項：`--ai` 用本機的 [Ollama](https://ollama.com) 模型來聊天和寫摘要（先執行 `ollama pull qwen3:1.7b`；電腦夠強可用
`--model qwen3:8b`）；`--tunnel` 另外印出公開的 Cloudflare 連結（需要 `cloudflared`）；`--no-grid` 不算全國格子；
`--port 8000`。如果模型檔載入失敗，請安裝 `models/nz_model.json` 裡寫的 scikit-learn 版本。
測試：`pip install -e ".[dev]"` 後執行 `pytest tests/nz`。

## 預測真的讓決策變好嗎？
在測試期每個地區的每一天（共 6,818 天），用同一套充電規則（`service.plan_charging`）搭配不同的預測，再用實際發電量結算。
電動車造成的電網用電，每天 kWh（越少越好）：

| 策略 | 電網用電（kWh/天） | 比「到了就充」少用 |
|---|---|---|
| 到了就充（08:00） | 18.01 | - |
| 每天固定 10:00–14:00 充（好習慣） | 8.46 | 9.55 |
| Persistence 預測 | 7.77 | 10.24 |
| 晴空 Persistence 預測 | 7.60 | 10.41 |
| **SolarCast v4 預測** | **6.96** | **11.05** |
| 完美預測（上限） | 5.31 | 12.70 |

和「固定 10–14 點充電」這個強基準比，SolarCast 每天少用 1.5 kWh（每間加工店每天約省 NZD 0.45、0.15 kg CO2）；
就算預測完美，也只能再多省 1.65 kWh/天。儀表板會同時顯示今天相對「到了就充」和「固定 10–14 點」的節省。
重現方式：`python experiments/nz_decision_value.py`（結果在 `experiments/results/nz_decision_value.md`）。

## 限制（請務必閱讀）
- **是模擬天氣，不是真實電廠。** 紐西蘭沒有公開的單一太陽能電廠逐時發電資料，所以我們使用 Open-Meteo 的模擬
  （再分析）日照資料，並用明確的公式換算發電量。模擬資料比真實情況平滑，真實電廠會比較難預測。
- **即時模式**使用 Open-Meteo 的預報服務（以及它對下一小時的前一天預報），來源和訓練資料略有不同，即時準確度可能稍低。
- **加工店和電動車是假設情境**（用電量、充電器、電價、碳排係數），都寫在程式碼裡。家用電動車估算使用常見的公開規格
  （Model Y RWD 約 60 kWh、交流 11 kW；Atto 3 60.48 kWh、交流 7 kW）、家裡 0.5 kW 用電、10% 充電損耗，
  以及最小充電功率 1.4 kW（`src/solarcast/nz/ev.py`）；實際車輛會有差異。
- **全國格子**的每一格借用最近地區的氣候代號，準確度只在 14 個地區量測過；每次更新約用掉 270 次免費的 Open-Meteo
  呼叫，所以回放其他日子每小時有次數限制。
- **AI 聊天還是可能答錯。** 小型語言模型可能誤讀資料，任何安全過濾器也都不完美。沒有 GPU 時回答很慢；
  如果模型載入失敗，聊天會改用固定範本回答。

## Repo 結構
```
src/solarcast/nz/   config、data、features、ensemble、model、service、grid、webapp、app、assistant、llm_setup、__main__
notebooks/          評審用與訓練用 notebook，英文版與繁體中文版
models/             訓練好的模型（nz_model.joblib）與報告（nz_model.json）
data/nz/            儀表板用的精簡回放資料（Open-Meteo，CC BY 4.0）
data/nz/grid_cells.json  全國格子的 137 個陸地格
docs/index.html     專案網站（GitHub Pages）
scripts/            train_nz.py、build_replay_data.py、build_grid_cells.py、make_nz_notebooks.py
tests/nz/           pytest 測試（離線）
```
團隊 Colab 操作說明：[docs/COLAB_GUIDE.zh-TW.md](docs/COLAB_GUIDE.zh-TW.md)。

## 資料與工具來源
天氣資料：Open-Meteo（CC BY 4.0）。函式庫：scikit-learn、pandas、NumPy、Flask。地圖：Leaflet 與 OpenStreetMap
貢獻者。圖表：Chart.js。公開連結：Cloudflare `cloudflared`。聊天：Qwen3（Apache-2.0）與 Llama Guard 3
（Llama 3.2 Community License；Built with Llama），用 Ollama 執行。AI 工具：Claude Code 協助撰寫程式、在 Colab 上跑實驗
與起草文件；團隊已檢查所有內容，並能解釋每個部分。完整清單：[NOTICE.zh-TW.md](NOTICE.zh-TW.md)。

本專案為原型，不構成能源或財務建議。

## 授權
[MIT 授權](LICENSE)：可以使用、複製、修改和分享程式碼、notebook、文件與訓練好的模型，只要保留版權聲明。
第三方資料、模型與函式庫依各自的授權（例如 Open-Meteo 資料是 CC BY 4.0，需要標示來源），詳見 [NOTICE.zh-TW.md](NOTICE.zh-TW.md)。
