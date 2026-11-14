# SolarCast NZ

[English](README.md) | **繁體中文**

紐西蘭下一小時太陽能發電預測，附互動地圖儀表板與電動車智慧充電決策。
Veritas AI 國際高中黑客松 2026，**AI + Climate** 賽道。

**評審請從這裡開始** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb)
Runtime > **Run all**（notebook 會要求免費的 T4 GPU，只給 AI 聊天用）。大約三分鐘內，notebook 會從本 repo 載入
我們訓練好的模型、顯示它的準確度，並印出儀表板的公開連結（免費的 Cloudflare tunnel）；AI 聊天再過幾分鐘就會準備好。

| Notebook | 用途 | English | 繁體中文 |
|---|---|---|---|
| 評審用 | 載入訓練好的模型、顯示結果、開啟儀表板 | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) |
| 訓練用 | 抓資料、訓練並評估模型、把模型備份到 GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) |

## 使用者與決策
一間紐西蘭的小加工店，屋頂有 50 kW 的太陽能板，還有一台每個工作天大約要充 40 kWh 的電動車。
老闆每小時都要決定：*現在充電，還是等更多陽光？* 在太陽能有剩的時候充電，用的是免費又乾淨的電；
時間選錯，就得向電網買電。

**一句話介紹：** *SolarCast 告訴紐西蘭的店家，什麼時候自家屋頂的陽光可以幫電動車充電。*

## 儀表板
- **紐西蘭地圖**：14 個地區，依「下一小時預測發電量」上色。
- **滑鼠移到地區上**：顯示當下天氣，以及有沒有在發電、發多少。
- **點擊地區**：顯示當天實際與預測的發電曲線、下一小時的電動車充電建議與原因，以及當天省下的電（度）、錢（紐幣）和 CO₂（公斤）。
- **回放**測試期間的任一個小時，或切換到**即時**（紐西蘭時間；晚上太陽能為 0）。
- **模型分頁**：我們模型和兩個基準方法的準確度比較、各地區表現、一週範例，以及模型最依賴哪些因素。
- **Ask SolarCast AI**（右下角）：用中文或英文解釋資料的聊天，例如「基督城的加工店現在該幫電動車充電嗎？」。
  它使用開源模型（Colab T4 GPU 上跑 Qwen3 8B，只有 CPU 時跑 Qwen3 1.7B），**只在 notebook 那台機器上本機執行**：
  不能上網、沒有工具、不需要 API 金鑰。伺服器先用和地圖相同的程式查出數字，模型只負責解釋；關鍵字過濾、Llama Guard 3
  安全分類器和系統規則會拒絕離題或有害的要求；提問也有頻率限制。預測本身是我們自己的模型，聊天只是架在上面的介面。

## 運作方式
- **資料**：Open-Meteo 提供的紐西蘭 14 個城市逐時天氣，2022 年到 2026 年 11 月（CC BY 4.0）。
- **任務**：在第 *t* 小時結束時，預測第 *t+1* 小時的日照量（GHI），再換算成一套 50 kW 系統的發電量。
  輸入只用第 *t* 小時以前已知的資料，加上太陽位置（有自動測試確保沒有偷看未來）。
- **基準方法**：Persistence（下一小時 = 這一小時）與晴空 Persistence（維持這一小時的雲量程度）。
- **模型（v4）**：scikit-learn 的梯度提升樹，所有地區共用一個模型，參數只用驗證期挑選。
  它把兩個模型取平均：一個直接預測日照量，一個預測「雲多不多」（晴空指數，因為太陽的軌跡可以精確算出）。
  兩個模型也都參考 Open-Meteo 針對被預測那一小時的**前一天預報**（約 24 小時前發布，所以沒有偷看未來；
  2024 年起才有存檔，更早的資料是缺值）。
- **依時間切分**：2022–2024 年訓練、2025 年 1–6 月驗證、2025 年 7 月–2026 年 10 月測試（只報告一次）。

## 結果（測試期、白天時段、全部 14 個地區）

| 方法 | 平均誤差（50 kW 系統） | 誤差佔容量 |
|---|---|---|
| Persistence | 4.03 kW | 8.1% |
| 晴空 Persistence | 1.66 kW | 3.3% |
| SolarCast v1（只用到目前為止的天氣） | 1.19 kW | 2.4% |
| **SolarCast v4（目前版本）** | **1.12 kW** | **2.2%** |

SolarCast v4 在每一個地區都贏過兩個基準方法，也贏過 v1。學習曲線（針對 v1 量測）顯示，模型大約需要半年的歷史資料
才能贏過晴空 Persistence，超過一年後進步就很小。完整數據：`models/nz_model.json`、`experiments/results/nz_learning_curve.md`。

**v4 是怎麼選出來的（只看驗證期，門檻事先訂好：要比 v1 好超過 1 W/m2）：**
v2 更多訓練地點：沒有進步；v3 加入前一天預報：小幅進步（0.71 W/m2），不夠；v4 預報輸入加上晴空指數模型再和原始模型取平均：
28.98 對 30.35 W/m2（進步 1.37），升級。細節：`experiments/results/nz_v2.json`、`nz_v3.json`、`nz_v4.json`。
測試期只看了一次，而且只針對 v4。

## 限制（請務必閱讀）
- **是模擬天氣，不是真實電廠。** 紐西蘭沒有公開的單一太陽能電廠逐時發電資料，所以我們使用 Open-Meteo 的模擬
  （再分析）日照資料，並用明確的公式換算發電量。模擬資料比真實情況平滑，真實電廠會比較難預測。
- **即時模式**使用 Open-Meteo 的預報服務（以及它對下一小時的前一天預報），來源和訓練資料略有不同，即時準確度可能稍低。
- **加工店和電動車是假設情境**（用電量、充電器、電價、碳排係數），都寫在程式碼裡。
- **AI 聊天還是可能答錯。** 小型語言模型可能誤讀資料，任何安全過濾器也都不完美。沒有 GPU 時回答很慢；
  如果模型載入失敗，聊天會改用固定範本回答。

## Repo 結構
```
src/solarcast/nz/   config、data、features、ensemble、model、service、webapp、app、assistant、llm_setup
notebooks/          評審用與訓練用 notebook，英文版與繁體中文版
models/             訓練好的模型（nz_model.joblib）與報告（nz_model.json）
data/nz/            儀表板用的精簡回放資料（Open-Meteo，CC BY 4.0）
scripts/            train_nz.py、build_replay_data.py、make_nz_notebooks.py
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
保留所有權利；允許為黑客松評審而檢視與執行。詳見 [LICENSE](LICENSE)。第三方資料與函式庫依其各自授權，
詳見 [NOTICE.zh-TW.md](NOTICE.zh-TW.md)。
