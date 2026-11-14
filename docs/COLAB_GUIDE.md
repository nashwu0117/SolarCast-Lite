# Colab guide for the team

Everyone works in Google Colab. The code lives in the GitHub repo
`nashwu0117/SolarCast-Lite` (currently **private**). Read this once, then keep it open.

**Deadline: Sunday Nov 15, 2026, 11:59 PM ET (= Monday Nov 16, 12:59 Taiwan time).**

## 1. Get access to the private repo (one time)

1. Send your GitHub username to the repo owner and accept the **collaborator invitation** (check your email).
2. Open any notebook link below. Colab will ask to connect to GitHub: click **Authorize** (allow access to private repos).
3. Create a token so notebooks can clone and push:
   - GitHub, Settings, Developer settings, Personal access tokens, **Fine-grained tokens**, Generate new token.
   - Repository access: *Only select repositories*, choose `SolarCast-Lite`.
   - Permissions: **Contents: Read and write**.
   - Copy the token (you only see it once).
4. In Colab, click the key icon (**Secrets**) in the left sidebar, **Add new secret**:
   name `GITHUB_TOKEN`, value = your token, and switch **Notebook access** on.
   Never paste the token into a notebook cell or chat.

## 2. Open a notebook

Open the link (replace nothing, just click):

| Notebook | Link |
|---|---|
| main (submission) | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/main.ipynb |
| 01 data exploration | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/01_data_exploration.ipynb |
| 02 modeling | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/02_modeling.ipynb |
| 03 demo | https://colab.research.google.com/github/nashwu0117/SolarCast-Lite/blob/main/notebooks/03_demo.ipynb |

Or: Colab, File, Open notebook, GitHub tab, pick `nashwu0117/SolarCast-Lite`.

Run the first code cell (setup) first. It clones the repo into `/content/SolarCast-Lite` (or pulls the latest),
adds `src/` to the Python path and turns on autoreload. Then run the rest top to bottom.

## 3. Save notebook changes to GitHub

File, **Save a copy in GitHub**. Pick repo `nashwu0117/SolarCast-Lite`, branch **main**, the same file path,
write a short commit message (for example `02: try deeper trees on val`), and tick the box to include the Colab link if you like.
Do not save a notebook someone else owns without asking first.

## 4. Edit the Python files (`src/solarcast/*.py`)

1. Open the file browser (folder icon on the left) and go to `/content/SolarCast-Lite/src/solarcast`.
2. Double-click a file to edit it in Colab. Save with Ctrl/Cmd+S.
3. Autoreload is on, so the next time you run a cell that uses it, the new code is used. If something looks stale, Runtime, Restart session, and re-run the setup cell.

### Push your edits to GitHub

Paste this into a **new cell** at the bottom of your notebook (never commit it into the notebook with the token filled in; it reads the token from Secrets). Fill in your name and email and the commit message:

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
git("add", "src", "tests", "docs", "README.md", "notebooks", "scripts")
git("commit", "-m", MESSAGE)
git("pull", "--rebase", url, "main")
git("push", url, "HEAD:main")
```

If `pull --rebase` reports a conflict, stop and ask the owner of that file. The token is never printed.

## 5. Get your teammates' changes

Re-run the setup cell (it runs `git pull --ff-only`). Then Runtime, **Restart session** if code looks out of date.
If the pull says it cannot fast-forward, you have local edits: push them first (section 4), then pull.

## 6. Ownership rules

**One person edits a given file at a time.** Before editing a file you do not own, tell its owner.

| Role | Owns |
|---|---|
| Data Lead | `src/solarcast/data.py`, `features.py`, `baselines.py`, `notebooks/01_data_exploration.ipynb` |
| ML Lead | `src/solarcast/models.py`, `evaluate.py`, `pipeline.py`, `notebooks/02_modeling.ipynb` |
| Product & Demo Lead | `src/solarcast/demo.py`, `notebooks/03_demo.ipynb`, `notebooks/main.ipynb`, `README.md` |

Shared and changed only by agreement: `src/solarcast/config.py` and `docs/CONTRACT.md`. If an interface changes, update
`docs/CONTRACT.md` first and tell everyone.

Golden rule: **never tune on the test split.** Train on `train`, choose on `val`, report `test` once.

## 7. Run the tests in Colab

```python
!pip -q install pytest
!cd /content/SolarCast-Lite && python -m pytest -q
```

(If the second line fails on the path, run `%cd /content/SolarCast-Lite` first, then `!python -m pytest -q`.)

## 8. Before submitting

1. In `main.ipynb`: Runtime, **Restart session and run all**. Everything must run with no errors.
2. Check the demo launches and prints a public link.
3. Save the notebook with outputs to GitHub (section 3) and make sure the README results section is filled in with real numbers.
4. Submit before **Sun Nov 15, 11:59 PM ET (Mon Nov 16, 12:59 Taiwan time)**. Aim to finish a few hours early.
