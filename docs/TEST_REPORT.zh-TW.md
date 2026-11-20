# SolarCast NZ 測試報告（2026-11-15）

環境：macOS、Python 3.14、pandas 3.1.1、numpy 2.5.4；乾淨 venv，scikit-learn 先用 1.9.1，再換成模型存檔時的 1.6.1。
測試基準 commit：`cd2bd8a`（之後 flask 依賴與 MIT 授權已另外提交）。

## 1. 自動化測試：全數通過
| 項目 | 結果 |
|---|---|
| `pytest`（含 Flask） | **102 passed, 0 failed, 0 skipped**（約 27 秒） |
| 第一次（沒裝 Flask） | 72 passed, 3 skipped：`test_assistant`、`test_webapp`、`test_grid` 的網頁部分被跳過 |

發現：原本 `pyproject.toml` 沒宣告 `flask`，乾淨環境會直接跳過網頁測試。此問題已修正：`pyproject.toml` 現在宣告 `flask>=2.3`。

## 2. 獨立重現模型成績（用 repo 內的 replay 資料與已存模型重算）
測試期 2025-07 至 2026-10，白天，14 區，73,746 筆。

| 方法 | 重現 MAE | 存檔數字 | 是否一致 |
|---|---|---|---|
| SolarCast v4 | 29.55 W/m²（1.122 kW） | 1.121 kW | 是 |
| Persistence | 105.44 W/m²（4.034 kW） | 4.034 kW | 是 |
| Clear-sky persistence | 43.82 W/m²（1.662 kW） | 1.662 kW | 是 |
| 只用 Open-Meteo 日前預報（新增基線） | 49.25 W/m²（1.878 kW） | — | 模型比它好 40% |

- 14 個區域全部贏過 clear-sky 基線（14/14），與 README 一致。
- 相對 clear-sky 基線的進步為 **32.6%**，相對 persistence 為 72.0%。
- 少雲（雲量 ≤60%）誤差 15.8 W/m²，多雲 41.1 W/m²，多雲時誤差約 2.6 倍。
- 準確率口徑：README 的 97.8% 是以 50 kW 裝置容量為分母。若改用白天平均發電量（14.27 kW）當分母，MAE 約為 **7.9%**。兩種算法都正確，但答辯時要先講清楚口徑。

## 3. EV 充電決策價值（新實驗，6,818 個區域日）
用同一個 `plan_charging` 規則，只換預報來源，以實際發電量結算 EV 造成的電網用電（kWh/日）：

| 策略 | 電網用電 | 比「到達就充」少用 |
|---|---|---|
| 到達就充（dashboard 的比較基準） | 18.01 | — |
| 固定 10–14 點充電 | 8.46 | 9.55 |
| Persistence 預報 | 7.77 | 10.23 |
| Clear-sky 預報 | 7.60 | 10.41 |
| **SolarCast v4 預報** | **6.96** | **11.05** |
| 完美預報（上限） | 5.31 | 12.69 |

- SolarCast 比固定中午充電少用 **1.5 kWh/日**（約 NZ$0.45、0.15 kg CO₂），比 clear-sky 預報少 0.64 kWh/日。
- 離完美預報還差 1.65 kWh/日，預報再進步的空間很有限。
- Dashboard 顯示的「節省」是對比「到達就充」，這個基準偏弱；對比固定中午充電，節省只有上面那個較小的數字。
- 此實驗的腳本是臨時分析，沒有放進 repo。

## 4. 其他檢查
| 項目 | 結果 |
|---|---|
| 模型與 scikit-learn 版本 | 存檔為 1.6.1，版本不同時載入會失敗；Judges 筆記本會安裝 1.6.1，已有處理 |
| `experiments/pooled_and_tuning.py` | 程式明確丟掉 test split，只用驗證期評分，未動到測試集 |
| 資料 | replay 共 165,312 列，預報欄位無缺值 |
| 警告 | 219 個 `Pandas4Warning`（`dayofyear` 已棄用），目前不影響，未來 pandas 會壞，改用 `day_of_year` 即可 |

## 5. 沒有測試到的部分
- 瀏覽器中的 dashboard 畫面與互動
- AI 聊天（需要 Ollama 與 GPU）
- Colab Judges 筆記本端到端、Cloudflare tunnel
- 即時模式（需連線 Open-Meteo）

## 6. 建議
1. （已完成）flask 依賴與 MIT 授權已提交，README 與 LICENSE 一致。
2. 簡報主打「比 clear-sky 基線好 32.6%」，並先說明 97.8% 的算法。
3. 把第 3 節的充電對比放進 README，並加上固定中午充電這個基準，數字誠實又有說服力。
4. 比賽前在乾淨的 Colab 跑一次 Judges 筆記本，並準備錄影備援。
