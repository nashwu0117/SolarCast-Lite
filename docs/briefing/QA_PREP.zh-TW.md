# SolarCast NZ：決賽問答準備

[English](QA_PREP.md) | **繁體中文**

決賽流程：簡報 3 分鐘（檔案 `SolarCast_NZ_Finals_3min.pptx`，講稿在備忘稿裡），接著 2 分鐘問答。
下面所有數字都來自 `README.md`、`docs/MODEL.md` 和 `experiments/results/`。

## 先搞懂整個專案（白話版）
1. **問題**：有屋頂太陽能的人（5 kW 的家庭到 50 kW 的工廠）想在「真的有太陽」的時候開 EV 充電、熱水器、熱泵。雲讓下一個小時很難猜。
2. **我們做的**：用 AI 預測「下一個小時，我的太陽能板會發多少 kW」，全紐西蘭任何地點都能查。
3. **資料**：Open-Meteo 的每小時天氣（2022 到 2026 年 11 月，14 個地區）。因為紐西蘭沒有公開的真實電廠逐時發電資料，所以「答案」是模型化（再分析）的日照，不是屋頂電表。
4. **模型**：梯度提升樹（scikit-learn）。v4 是兩個模型平均：一個直接預測日照，一個預測「晴空指數」（會有多少雲），再用天文學算出的晴空日照換回日照。另外也讀了氣象服務前一天發布的預報。
5. **評估怎麼誠實**：依時間切分（訓練 2022-2024、驗證 2025 上半年、測試 2025 年 7 月到 2026 年 10 月），不隨機切；所有設計都只看驗證集；測試集每個模型只看一次；升級門檻事先定好（要比舊版好超過 1 W/m2）。
6. **結果**：50 kW 系統平均誤差 1.12 kW（約佔容量 2.2%），比最強的簡單基準（晴空持續）少 32.6% 誤差，14 個地區全部贏過兩個基準。
7. **應用**：同一套預測驅動 EV 充電建議。6,818 個地區日測試：到家就充 18.01 kWh/天用電網，固定 10:00-14:00 充 8.46，SolarCast 6.96，完美預報 5.31。

## 要背的數字
| 項目 | 數值 |
|---|---|
| 預測對象 | 下一小時日照（GHI），再換算成 kW |
| 地區 / 網格 | 14 個地區；全國 137 格（0.5 度） |
| v4 測試誤差 | 29.6 W/m2 = 50 kW 系統 1.12 kW = 容量 2.2% = 平均日間發電 14.3 kW 的 7.9% |
| 對晴空持續 | 少 32.6%（基準 1.66 kW） |
| 對 Open-Meteo 預報 | 少 40.3%（基準 1.88 kW） |
| 對一般持續 | 少 72.2%（基準 4.03 kW） |
| 贏過基準的地區 | 14 / 14（最好 Christchurch 0.96 kW；最難 Whangarei 1.27、Invercargill 1.25） |
| EV 省電 | 比固定時段省 1.5 kWh/天，約 NZD 0.45、0.15 kg CO2 |
| 預測時間範圍 | 1 小時贏氣象服務 40%，2 小時 16%，3 小時 6%，6 小時起沒贏 |

儀表板 Model 分頁寫 29.5 W/m2（29.55 無條件捨去），文件寫 29.6，是同一個數字。

## 可能被問的問題與回答（英文可直接講）

**1. 你們預測什麼？**
EN: "Sunlight for the next hour, then power for the chosen panel size with a stated formula. Output scales linearly with panel size."
說明：預測日照，用公式換成 kW：`kW = 容量 × GHI/1000 × 0.80 × 溫度損失`。

**2. 用什麼模型？為什麼？**
EN: "Gradient-boosted decision trees. They suit tabular weather data, train in minutes on a CPU and handle missing values. v4 averages a direct sunlight model and a clear-sky-index model."
說明：很多小決策樹，後面的樹修正前面的錯。表格資料用它快又強。

**3. 為什麼不用深度學習或 LLM？**
EN: "For tabular data with a few years of history, trees are a strong, fast default. The LLM is only used to explain numbers in the chat; the forecast is our own model."

**4. 怎麼證明沒有偷看未來（資料洩漏）？**
EN: "We split by time, features only use data known at the end of hour t, and an automated test changes everything from t+1 onward to check that no input changes. The test period was looked at once."

**5. 97.8% 準確率是不是太高？**
EN: "It is 100% minus the error as a share of the 50 kW capacity. Against mean daytime output the error is 7.9%. The fairest number is 32.6% lower error than the strongest simple baseline."

**6. 基準是什麼？公平嗎？**
EN: "Persistence and clear-sky persistence, which is much harder to beat. We also compared with Open-Meteo's own day-ahead forecast."

**7. 資料是真的嗎？**
EN: "It is modelled reanalysis weather, because New Zealand has no public hourly output for individual solar plants. Real plants would show larger errors. That is our main limitation."
這題一定要老實講，簡報第 4、7 頁已經寫了。

**8. 有沒有用測試集調參數？**
EN: "No. All choices used validation only, with a bar fixed in advance of 1 W/m2. v2, v3, v5 and v6 did not pass the bar; v4 did. The test period was read once."

**9. 為什麼只預測 1 小時？**
EN: "We beat the weather service by 40% at 1 hour, 16% at 2, 6% at 3 and not from 6 hours. We stop where we add value."

**10. 需要多少歷史資料？**
EN: "About three months: 90 days of training already beats clear-sky persistence, and gains flatten after about one year."

**11. 哪裡會失敗？**
EN: "When clouds arrive or clear inside the hour, at dawn and dusk, and in cloudier southern regions like Invercargill and Queenstown."

**12. 地圖上任意一點是怎麼預測的？**
EN: "We download that spot's weather from Open-Meteo and run the same model, borrowing the nearest region's climate id. Accuracy was measured only at the 14 regions."

**13. EV 結果真實嗎？**
EN: "It is an illustration of the forecast's value. Car, charger, prices and CO2 factor are stated assumptions in the code."

**14. 為什麼只省 1.5 kWh/天？**
EN: "Because the 10:00-14:00 habit is already good, and a perfect forecast would save only 1.65 kWh/day more. Against charging on arrival the saving is 11.05 kWh/day."

**15. AI 聊天安全嗎？**
EN: "It runs an open model locally with no internet, no tools and no API keys. The server looks up the numbers, the model only explains them, and filters plus Llama Guard 3 refuse harmful requests. It can still be wrong."

**16. 對氣候有什麼幫助？**
EN: "Using more of your own sunshine means less grid energy and fewer emissions. Our measured example is small per site; the value is that anyone can use it for free."

**17. 你們怎麼用 AI 工具？**
EN: "Claude Code helped write code, run experiments and draft docs. The team reviewed everything and can explain every part."
你必須真的懂每個部分：把上面「先搞懂整個專案」讀熟。

**18. 能重現嗎？開源嗎？**
EN: "Yes, MIT license. The Colab and Kaggle Judges notebooks load our trained model and open the dashboard in about three minutes."

**19. 下一步？**
EN: "Compare with real rooftop output, extend beyond one hour if it adds value, and measure real EV savings."

**20. 最困難的是什麼？**
EN: "Keeping the evaluation honest: no future leakage, a fixed promotion bar, and reporting that more data did not help but new information did."

## 示範出狀況時
- 用 `docs/briefing/screenshots/` 裡的截圖（回放、EV 面板、Model 分頁），或網站 <https://nashwu0117.github.io/SolarCast-Lite/>。
- 本機執行：`pip install -e .`，再 `python -m solarcast.nz`，開 <http://127.0.0.1:8000>。如果模型載入失敗，裝 `models/nz_model.json` 指定的 scikit-learn 版本（1.6.1）。
- 晚上 Live 會顯示 0 kW，按 **Replay today's sun** 就能用白天資料示範。
