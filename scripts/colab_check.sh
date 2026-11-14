#!/usr/bin/env bash
# Upload the current working tree to a Colab session and run the tests and every notebook there.
#
# Usage: scripts/colab_check.sh [session-name]
# Requires the official Colab CLI (`uv tool install google-colab-cli`) authenticated with ADC, and an
# existing session (`colab --auth=adc new -s solarcast`).
#
# The job runs in the background on the VM (long websocket executions can drop), and this script polls a
# log file with short calls until the job writes its DONE marker.
set -euo pipefail
SESSION="${1:-solarcast}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$ROOT")"
TMP="$(mktemp -d)"
COLAB=(colab --auth=adc)

COPYFILE_DISABLE=1 tar czf "$TMP/repo.tar.gz" -C "$(dirname "$ROOT")" \
  --exclude="$NAME/.venv" --exclude="$NAME/data/raw" \
  --exclude='*.egg-info' --exclude='__pycache__' --exclude='.pytest_cache' "$NAME"

cat > "$TMP/check.sh" <<'REMOTE'
#!/usr/bin/env bash
cd /content && rm -rf SolarCast-Lite out && tar xzf repo.tar.gz --no-same-owner 2>/dev/null
cd SolarCast-Lite && mkdir -p /content/out
pip -q install pytest gradio nbconvert >/dev/null 2>&1
echo "pytest: $(PYTHONPATH=src python -m pytest -q 2>&1 | tail -1)"
for n in main 01_data_exploration 02_modeling 03_demo; do
  if SOLARCAST_LAUNCH_DEMO=0 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
       --output-dir /content/out "notebooks/$n.ipynb" > "/content/out_$n.log" 2>&1; then
    echo "$n: OK"
  else
    echo "$n: FAIL"; grep -E "^[A-Za-z]*Error" "/content/out_$n.log" | tail -3
  fi
done
echo DONE
REMOTE

"${COLAB[@]}" upload -s "$SESSION" "$TMP/repo.tar.gz" /content/repo.tar.gz >/dev/null
"${COLAB[@]}" upload -s "$SESSION" "$TMP/check.sh" /content/check.sh >/dev/null
printf '%s\n' '%%bash' 'rm -f /content/check.log; nohup bash /content/check.sh > /content/check.log 2>&1 &' \
  | "${COLAB[@]}" exec -s "$SESSION" --timeout 60 >/dev/null
echo "started on Colab session '$SESSION'; polling..."

for _ in $(seq 1 90); do
  sleep 20
  LOG="$(printf '%s\n' '%%bash' 'cat /content/check.log 2>/dev/null || true' \
    | "${COLAB[@]}" exec -s "$SESSION" --timeout 60 2>/dev/null || true)"
  if grep -q '^DONE' <<<"$LOG"; then
    grep -v '^DONE' <<<"$LOG"
    grep -q 'FAIL' <<<"$LOG" && exit 1 || exit 0
  fi
done
echo "timed out waiting for the Colab job"; exit 2
