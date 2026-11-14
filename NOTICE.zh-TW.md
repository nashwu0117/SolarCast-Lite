# 第三方授權聲明與資料來源

[English](NOTICE.md) | **繁體中文**

SolarCast NZ 自己的程式碼以 [MIT 授權](LICENSE)開源。以下內容屬於其他人，依各自的授權使用。
黑客松規則要求列出所有使用到的資料集、預訓練模型、教學與 AI 工具，這份檔案就是那份清單。

## 資料

| 內容 | 來源 | 授權 | 我們怎麼用 |
|---|---|---|---|
| Open-Meteo 歷史天氣 API（ERA5 再分析資料） | https://open-meteo.com/en/docs/historical-weather-api | CC BY 4.0 | 紐西蘭 14 個城市 2022 年到 2026 年 11 月的逐時天氣：訓練、驗證、測試。 |
| Open-Meteo 歷史預報 API（前一天預報存檔） | https://open-meteo.com/en/docs/previous-runs-api | CC BY 4.0 | v4 模型的前一天預報輸入（2024 年起）。 |
| Open-Meteo 預報 API | https://open-meteo.com/en/docs | CC BY 4.0 | 儀表板的即時模式與全國格子。 |
| Open-Meteo 海拔 API（Copernicus DEM） | https://open-meteo.com/en/docs/elevation-api | CC BY 4.0（DEM：Copernicus，(c) DLR/ESA） | 只用一次，挑出全國格子的陸地格（`data/nz/grid_cells.json`）。 |
| `data/nz/replay.csv.gz` | 上述 Open-Meteo 資料的一部分 | CC BY 4.0 | 儀表板的回放資料（只有測試期）。 |

CC BY 4.0 要求的姓名標示：*天氣資料來自 Open-Meteo.com（https://open-meteo.com），以 CC BY 4.0 授權
（https://creativecommons.org/licenses/by/4.0/）。我們從中衍生出特徵、太陽能發電估計、模型與評估結果。*

## 預訓練模型（只用在 SolarCast AI 聊天；太陽能預測是我們自己訓練的模型）

| 模型 | 來源 | 授權 | 我們怎麼用 |
|---|---|---|---|
| Qwen3 8B 與 Qwen3 1.7B（阿里巴巴 Qwen 團隊） | https://huggingface.co/Qwen/Qwen3-8B、https://ollama.com/library/qwen3 | Apache-2.0 | 儀表板聊天：用白話解釋我們的資料（有 GPU 用 8B，只有 CPU 用 1.7B）。沒有微調。 |
| Llama Guard 3 1B（Meta） | https://huggingface.co/meta-llama/Llama-Guard-3-1B、https://ollama.com/library/llama-guard3 | Llama 3.2 Community License | 檢查聊天問題的安全分類器。**Built with Llama.** |

## 軟體

| 軟體 | 授權 | 用途 |
|---|---|---|
| scikit-learn | BSD-3-Clause | 我們的預測模型（梯度提升） |
| pandas、NumPy | BSD-3-Clause | 資料處理 |
| Matplotlib | Matplotlib License（以 PSF 為基礎） | notebook 圖表 |
| Flask、Werkzeug | BSD-3-Clause | 儀表板網頁伺服器 |
| Leaflet | BSD-2-Clause | 地圖 |
| OpenStreetMap 地圖圖磚 | 資料 (c) OpenStreetMap 貢獻者，ODbL | 地圖底圖 |
| Chart.js | MIT | 儀表板圖表 |
| Cloudflare cloudflared | Apache-2.0 | 免費公開連結（quick tunnel） |
| Ollama | MIT | 在 Colab 機器上本機執行開源語言模型 |
| pytest、nbformat、nbconvert、Jupyter | MIT / BSD-3-Clause | 測試與執行 notebook |
| Google Colab CLI（google-colab-cli） | Apache-2.0 | 在 Colab 上執行所有程式 |

## AI 工具

- **Claude Code**（Anthropic）：用來撰寫與檢查程式碼、在 Colab 上跑實驗、起草文件。
  團隊已檢查過所有成果，並能解釋專案的每個部分。

## 假設（非第三方資料）

`src/solarcast/nz/config.py` 中的加工店用電、電動車充電需求、充電器功率、電價與電網碳排係數，
都是我們自己設定並寫明的假設，不是量測資料。
