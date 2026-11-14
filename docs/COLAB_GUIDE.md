# Colab guide for the team

**English** | [繁體中文](COLAB_GUIDE.zh-TW.md)

Everyone works in Google Colab; nothing needs to run on your own computer. The code lives in the **public** GitHub repo
`nashwu0117/SolarCast-Lite`. Read this once, then keep it open.

**Deadline: Sunday Nov 15, 2026, 11:59 PM ET (= Monday Nov 16, 5:59 PM New Zealand time).** Submit a first version a
few hours early, then resubmit if you improve it.

## 1. The two notebooks

| Notebook | What it does | English | 繁體中文 |
|---|---|---|---|
| **Judges** (submission) | Loads our trained model from GitHub, shows the test results, opens the map dashboard with the AI chat | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/SolarCast_NZ_Judges.zh-TW.ipynb) |
| **Train** | Downloads Open-Meteo history, trains and evaluates the model, backs it up to GitHub | [Open](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.ipynb) | [開啟](https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/train.zh-TW.ipynb) |

The **English Judges notebook is the one we submit.** The zh-TW one is the same code with Chinese explanations, for
testing and learning.

## 2. Run the Judges notebook

1. Open the link. It asks Colab for a free **T4 GPU** (Runtime > Change runtime type shows "T4 GPU"). If Colab says no
   GPU is available, choose **Connect without GPU**: everything still works, only the AI chat is slower.
2. Runtime > **Run all**.
3. After about 2-3 minutes, section 5 prints a big **dashboard link** (a free Cloudflare tunnel). Open it.
4. The **Ask SolarCast AI** button (bottom right of the dashboard) becomes a real AI chat a few minutes later, once the
   model is downloaded and loaded. Until then it answers with fixed templates. Section 6 waits for it and asks two test
   questions (one normal, one it must refuse).
5. Keep the notebook tab open while you use the link: the dashboard runs on that Colab machine.

Things to try on the dashboard: drag the **Replay** slider to a New Zealand daytime hour (in Live mode at night every
region shows 0 kW, which is correct), hover a region, click it for the detail panel, open the **Model** tab, and ask the
chat "Which region generates the most solar power next hour?" or「基督城今天可以省多少錢？」.

## 3. Run the Train notebook

- `QUICK = True` (config cell): a 1-minute smoke test with 2 regions x 120 days. The model in the repo is not touched.
- `QUICK = False`: the full run, 14 regions from 2022. The Open-Meteo download can take about 30 minutes because of
  rate limits.
- Backup to GitHub (last cell, `BACKUP = True`) needs a token. See section 6. Only promote a model that beats the current
  one on **validation** by more than 1 W/m2; the test period is looked at once per promoted model.

## 4. Change the code

- Python code: `src/solarcast/nz/` (config, data, features, ensemble, model, service, webapp, app, assistant,
  llm_setup). Open the file browser (folder icon), go to `/content/SolarCast-Lite/src/solarcast/nz`, double-click a file,
  save with Ctrl/Cmd+S, then Runtime > **Restart session** and run again (autoreload does not work on Colab's Python 3.13).
- **Notebooks are generated.** Do not edit the `.ipynb` files by hand: change `scripts/make_nz_notebooks.py` (code cells
  are shared, the text is written in English and zh-TW there) and run `python scripts/make_nz_notebooks.py`. That keeps
  the two languages identical.
- Module interfaces are written down in `docs/NZ_CONTRACT.md`. If you change one, update that file first and tell the team.

## 5. Run the tests in Colab

```python
!pip -q install pytest flask
!cd /content/SolarCast-Lite && PYTHONPATH=src python -m pytest -q tests/nz
```

All 57 tests should pass in about 20 seconds. They are offline: no model download, no network.

## 6. Push your changes to GitHub

1. Create a token (one time): GitHub > Settings > Developer settings > Personal access tokens > **Fine-grained tokens** >
   Generate new token. Repository access: only `SolarCast-Lite`. Permissions: **Contents: Read and write**.
2. In Colab, click the key icon (**Secrets**) > **Add new secret**: name `GITHUB_TOKEN`, value = your token, and turn
   **Notebook access** on. Never paste the token into a cell, a commit or a chat.
3. Paste this into a new cell (it reads the token from Secrets and never prints it):

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

Write commit messages in English in the Conventional Commits style (`feat(nz): ...`, `fix(nz): ...`, `docs: ...`). If
`pull --rebase` reports a conflict, stop and ask the person who owns that file.

## 7. Who owns what

**One person edits a given file at a time.** Before editing a file you do not own, tell its owner.

| Area | Files |
|---|---|
| Data and features | `src/solarcast/nz/data.py`, `features.py`, `data/nz/`, `scripts/build_replay_data.py` |
| Model and evaluation | `src/solarcast/nz/model.py`, `ensemble.py`, `scripts/train_nz.py`, `docs/MODEL*.md` |
| Dashboard and AI chat | `src/solarcast/nz/service.py`, `webapp.py`, `assistant.py`, `llm_setup.py`, `app.py` |
| Notebooks and write-up | `scripts/make_nz_notebooks.py`, `README*.md`, `NOTICE*.md`, the one-page write-up and the video |

Shared, changed only by agreement: `src/solarcast/nz/config.py` and `docs/NZ_CONTRACT.md`.

Golden rule: **never tune on the test period.** Train on 2022-2024, choose on Jan-Jun 2025, report Jul 2025-Oct 2026 once.

## 8. Before submitting

1. Open the **English Judges** link in an incognito window (it must open without signing in, so the repo must stay public) and
   Runtime > **Run all**. Everything must run with no errors, the dashboard link must open, and the chat must answer.
2. The notebook in the repo is saved **with outputs** (the brief requires it). If you change it, re-run it on Colab and
   save it with outputs.
3. Check the README results and the credits (`NOTICE.md`: every dataset, pretrained model and AI tool, with links).
4. Submission form: code link (the English Judges notebook or the repo), write-up link (README or a Google Doc set to
   "anyone with the link can view"), optional video (up to 2 minutes, unlisted YouTube or Drive), track **AI + Climate**,
   every teammate's name, email and school, and what we added ("Make it your own": New Zealand, next-hour instead of
   next-day, the EV charging decision, the AI chat).
5. Submit before **Mon Nov 16, 5:59 PM New Zealand time**, ideally a few hours earlier.
