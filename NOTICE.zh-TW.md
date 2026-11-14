# 第三方授權聲明與資料來源

[English](NOTICE.md) | **繁體中文**

SolarCast Lite 自己的程式碼適用 [LICENSE](LICENSE)（保留所有權利）。以下內容屬於其他人，依各自的授權使用。
黑客松規則要求列出所有使用到的資料集、模型、教學與 AI 工具，這份檔案就是那份清單。

## 資料

| 內容 | 來源 | 授權 | 我們怎麼用 |
|---|---|---|---|
| Two-Site Solar PV Inverter & Weather Dataset（羅馬尼亞） | https://github.com/upb-nlp/PV-Generation-Dataset | CC BY 4.0 | 執行時才下載（不在本 repo 內重新散布）。使用 Site A、Site B 的逐時太陽能發電、天氣與太陽位置；實驗中也用了其中的歷史天氣預報。 |
| Open-Meteo 天氣資料（歷史資料與歷史預報） | https://open-meteo.com，已包含在上述資料集中 | CC BY 4.0 | 透過資料集取得的天氣特徵。 |
| 太陽位置 | pvlib（https://pvlib-python.readthedocs.io），由資料集作者計算 | BSD-3-Clause | 透過資料集取得的太陽高度角與方位角欄位。 |

CC BY 4.0 要求的姓名標示：*「Two-Site Solar PV Inverter & Weather Dataset (Romania)」，作者為 upb-nlp 資料集作者群，
以 CC BY 4.0 授權（https://creativecommons.org/licenses/by/4.0/）。我們從中衍生出特徵、模型與評估結果；原始資料未經修改。*
建議的引用格式請見資料集的 `CITATION.cff`。

## 軟體函式庫

| 函式庫 | 授權 |
|---|---|
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Matplotlib | Matplotlib License（以 PSF 為基礎） |
| Gradio | Apache-2.0 |
| pytest、nbformat、nbconvert、Jupyter | MIT / BSD-3-Clause |
| Google Colab CLI（google-colab-cli），用來在 Colab 上執行所有程式 | Apache-2.0 |

## AI 工具

- **Claude Code**（Anthropic）：用來撰寫與檢查程式碼、在 Colab 上跑實驗、起草文件。
  團隊已檢查過所有成果，並能解釋專案的每個部分。

## 假設（非第三方資料）

`src/solarcast/charging.py` 中的加工店用電曲線、電動車充電需求、電價與電網碳排係數，
都是我們自己設定的合成假設，不是量測資料。
