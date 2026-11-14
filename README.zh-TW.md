# SolarCast Lite

[English](README.md) | **繁體中文**

用 scikit-learn 預測下一小時的太陽能發電功率，並提供互動式回放 Demo。
Veritas AI 國際高中黑客松 2026，**AI + Climate** 賽道。

**評審請從這裡開始** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_Lite_Submission.ipynb)：`notebooks/SolarCast_Lite_Submission.ipynb` 不需要其他檔案就能執行。Runtime > Run all （免費 CPU 環境約 15 秒）。在最後一格勾選 `RUN_DASHBOARD` 即可開啟互動儀表板（Cloudflare 公開連結）。

團隊用 notebook：

| Notebook | 在 Colab 開啟 |
|---|---|
| main（交件用） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/main.ipynb) |
| 01 資料探索 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 建模 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/02_modeling.ipynb) |
| 03 Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/03_demo.ipynb) |

## 專案目標（請先讀這段）

**使用者：** 一間小加工店的老闆，屋頂裝了 50 kW 的太陽能板，有一台電動車（自用車或送貨用的貨車），
每個工作天大約需要充 40 kWh 的電。

**他每小時要做的決定：**「現在要充電，還是晚點再充？」在太陽能板發電有剩的時候充電，用的是免費又乾淨的太陽能；
時間選錯，就得向電網買電。

**我們要做的：**
1. **SolarCast 預測模型**：用最近的發電量、太陽位置和天氣，預測下一小時的太陽能發電量，並且要贏過簡單的基準方法。
2. **智慧充電決策**：每小時用「預測發電量 − 店裡自己的用電」算出預估剩餘電量；剩得夠多就幫電動車充電，
   而且傍晚前一定要充滿。
3. **證明它有用**：在保留的測試期間（2026 年 3–6 月）一天一天回放，比較四種充電策略：
   一到店就充（不看預測）、Persistence 預測、**SolarCast**、完美預知（理論上限）。
   報告少買了多少度電、省了多少錢、少排了多少 CO₂。

**一句話介紹：** *SolarCast 告訴小店家什麼時候該幫電動車充電，讓車子用自家屋頂的陽光跑，而不是用電網的電。*

**哪些是真實資料、哪些是假設：** 太陽能發電量和天氣是羅馬尼亞兩座 50 kW 電廠的真實量測資料。
加工店的用電曲線和電動車的充電需求是我們假設的合成資料，會在 notebook 裡寫清楚。

```
預測下一小時發電  ->  扣掉店裡用電  ->  預估剩餘電量
                                       |-- 夠多：現在幫電動車充電
                                       `-- 不夠：等一下（但傍晚前要充滿）
```

## 進度

| 項目 | 狀態 |
|---|---|
| 資料流程、不偷看未來的特徵、基準方法 | 完成，30 個測試在 Colab 上全部通過 |
| 預測模型 A（發電量 + 太陽位置）和 B（再加天氣） | 完成；在測試集上，Model B 的誤差約為 Persistence 的一半 |
| 實驗：天氣預報特徵、晴空指標、合併訓練與調參數 | 結果在 `experiments/results/`（只用驗證集） |
| 電動車充電決策模擬 | 完成；結果在交件 notebook 第 5 節 |
| 給評審的交件 notebook、用 Cloudflare tunnel 開的儀表板 | 完成，已在 Colab 驗證 |
| 一頁書面報告和 2 分鐘 demo 影片 | 團隊成員負責 |

## 問題
太陽能電廠的價值取決於它有多「可預測」。雲層會讓發電量快速變化，電網調度員和電廠業主需要準確掌握下一小時的發電量，
才能平衡供需、安排可調節的用電與儲能。更好的短期預測，能讓太陽能取代更多化石燃料發電。電動車是最大的可調節用電之一：
如果在對的時間充電，就能把原本會浪費或低價賣掉的太陽能電用掉，也不必向電網買電。

## 方法
- **任務：** 在第 `t` 小時結束時，預測一座 50 kW 電廠第 `t+1` 小時的 `active_power_kw`。輸入只用到 `t` 以前的資料，
  加上目標小時的太陽位置與日曆資訊（不會用到未來資料，詳見 `docs/CONTRACT.md`）。
- **基準模型：** Persistence（下一小時 = 這一小時）與 Previous-day same hour（前一天同一小時）。
- **Model A：** 只用發電量歷史與太陽幾何的梯度提升模型。
- **Model B：** Model A 再加上預測當下的天氣（雲量、日照輻射、溫度等）。
- **評估方式：** 依時間切分（2025 年訓練、2026 年 1–2 月驗證、2026 年 3–6 月測試），只看白天時段，
  指標為 MAE、RMSE，以及 MAE 佔裝置容量的百分比；並依天氣狀況與發電量區間拆解誤差，另做跨電廠測試（A 訓練、B 測試）。
- **資料：** Two-Site Solar PV Inverter & Weather Dataset（羅馬尼亞），每小時資料。

## Repo 結構
```
src/solarcast/   config、data、features、baselines、models、evaluate、pipeline、demo
notebooks/       main.ipynb（交件用）與 01/02/03 練習用 notebook（用 scripts/make_notebooks.py 重新產生）
tests/           pytest 測試（離線、使用合成資料）
docs/            CONTRACT.md（模組介面規格）、COLAB_GUIDE.md（團隊 Colab 操作說明）
scripts/         make_notebooks.py
```

## 本機環境設定
```bash
uv venv --python 3.12 && uv pip install -e ".[demo,dev]"
.venv/bin/python -m pytest -q
.venv/bin/jupyter lab
```

## Colab 設定
每本 notebook 的第一格都是 setup cell，會 clone repo 並把 `src/` 加進 Python 路徑。私有 repo 的存取權限、
如何儲存修改與團隊規則，請看 [docs/COLAB_GUIDE.zh-TW.md](docs/COLAB_GUIDE.zh-TW.md)。在 Colab 跑測試：
`!cd /content/SolarCast-Lite && PYTHONPATH=src python -m pytest -q`。

## 結果
待辦：最終版 `main.ipynb` 在 Colab 跑完後填入（Model B、Model A 與兩個基準模型的測試集 MAE / nMAE；
不同天氣狀況的誤差；跨電廠結果）。

## 限制
- **歷史回放。** Demo 是在保留的歷史時段上回放，不是即時預測服務。
- **再分析天氣資料。** 天氣輸入是 Open-Meteo 再分析資料（實際發生的天氣），不是天氣預報，
  所以實際部署時加上預報誤差後，準確度會比較低。
- **單一氣候區。** 只有兩座羅馬尼亞的電廠、約 1.5 年資料，結果不一定適用於其他地區。

## 資料與工具來源
- 資料集：*Two-Site Solar PV Inverter & Weather Dataset (Romania)*，https://github.com/upb-nlp/PV-Generation-Dataset，CC BY 4.0
  （內含 Open-Meteo 天氣資料）。
- 太陽位置：pvlib。函式庫：scikit-learn、pandas、matplotlib、Gradio。
- AI 工具：Claude Code。

本專案為原型（prototype）。

## 授權
保留所有權利；允許為黑客松評審而檢視與執行。詳見 [LICENSE](LICENSE)。第三方資料與函式庫依其各自授權，詳見 [NOTICE.zh-TW.md](NOTICE.zh-TW.md)。
