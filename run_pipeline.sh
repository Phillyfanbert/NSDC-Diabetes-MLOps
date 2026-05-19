#!/bin/bash
# run_pipeline.sh — full MLOps pipeline + serving in one command
# ==============================================================
# Usage: bash run_pipeline.sh
#
# Runs the full pipeline in order:
#   1.  Create logs directory
#   2.  Start MLflow UI (background)
#   3.  Fetch data from WHO API
#   4.  Clean raw data
#   5.  Validate clean data        ← exits with code 1 if hard checks fail
#   6.  Engineer features + save scale_params.json
#   7.  Train both models & register in MLflow
#   8.  Start FastAPI server (background)
#          └─ serve_model.py runs promotion logic here, Ridge wins if within tolerance
#   9.  Verify API health + confirm which model was promoted
#   10. Run analysis scripts against the promoted Production run
#   11. Generate dashboard HTML
#   12. Open dashboard in browser
#
# Press Ctrl+C once to shut everything down cleanly.

set -e   # exit immediately if any command fails

# ---------------------------------------------------------------------------
# Cleanup: kill background servers on exit (Ctrl+C or pipeline error)
# ---------------------------------------------------------------------------
MLFLOW_PID=""
API_PID=""

cleanup() {
  echo ""
  echo "Shutting down servers..."
  [ -n "$MLFLOW_PID" ] && kill "$MLFLOW_PID" 2>/dev/null
  [ -n "$API_PID"    ] && kill "$API_PID"    2>/dev/null
  echo "Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════╗"
echo "║   NSDC Diabetes MLOps Pipeline       ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create logs directory FIRST (before anything tries to write to it)
# ---------------------------------------------------------------------------
echo "1. Setting up logs directory..."
mkdir -p logs
echo "   ✅ logs/ ready"

# ---------------------------------------------------------------------------
# Step 2: Start MLflow UI BEFORE pipeline scripts run
#         Every script logs to MLflow, so it must be up first.
# ---------------------------------------------------------------------------
echo ""
echo "2. Starting MLflow UI at http://127.0.0.1:5000 ..."
mlflow ui --host 127.0.0.1 --port 5000 > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!
echo "   Waiting for MLflow to be ready..."
for i in $(seq 1 15); do
  if curl -s http://127.0.0.1:5000 > /dev/null 2>&1; then
    echo "   ✅ MLflow UI ready (after ${i}s)"
    break
  fi
  sleep 1
  if [ "$i" -eq 15 ]; then
    echo "   ⚠️  MLflow didn't respond in 15s — continuing anyway"
  fi
done
echo "   ✅ MLflow UI running (PID $MLFLOW_PID)"

# ---------------------------------------------------------------------------
# Step 3: Fetch data from WHO API (with retry logic)
# ---------------------------------------------------------------------------
echo ""
echo "3. Fetching data from WHO API..."
python src/fetch_data.py

# ---------------------------------------------------------------------------
# Step 4: Clean raw data
#         Must run BEFORE validation — we validate the clean file, not raw.
# ---------------------------------------------------------------------------
echo ""
echo "4. Cleaning raw data..."
python src/cleaning.py

# ---------------------------------------------------------------------------
# Step 5: Validate clean data
#         Will exit with code 1 if any hard check fails, stopping the pipeline.
# ---------------------------------------------------------------------------
echo ""
echo "5. Validating clean data..."
python src/validate_data.py

# ---------------------------------------------------------------------------
# Step 6: Feature engineering — also saves scale_params.json
# ---------------------------------------------------------------------------
echo ""
echo "6. Engineering features..."
python src/features.py

# ---------------------------------------------------------------------------
# Step 7: Train both models and register in MLflow registry
# ---------------------------------------------------------------------------
echo ""
echo "7. Training models (Linear Regression + Ridge)..."
python src/train_model.py

# Kill any process still holding port 8000 from a previous run
echo "   Clearing port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 8: Start FastAPI server — promotion happens here at startup
#         serve_model.py runs promote_best_model() which applies the Ridge
#         preference logic and tags the winning model as Production.
#         Analysis scripts in Step 10 must run AFTER this so they query
#         the correct Production run.
# ---------------------------------------------------------------------------
echo ""
echo "8. Starting API server at http://127.0.0.1:8000 ..."
echo "   (Promotion logic runs here — Ridge preferred if within tolerance)"
python src/serve_model.py > logs/api.log 2>&1 &
API_PID=$!
echo "   Waiting for API to promote model and be ready..."
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "   ✅ API server ready (after ${i}s)"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "   ⚠️  API didn't respond in 30s — check logs/api.log"
  fi
done
echo "   ✅ API server running (PID $API_PID)"

# ---------------------------------------------------------------------------
# Step 9: Verify API health and confirm which model was promoted
#         This must appear BEFORE analysis scripts so you can see the
#         promotion decision in the terminal output.
# ---------------------------------------------------------------------------
echo ""
echo "9. Verifying API health and promotion decision..."
if command -v curl &>/dev/null; then
  HEALTH=$(curl -s http://127.0.0.1:8000/health 2>/dev/null || echo "unreachable")
  echo "   $HEALTH"

  # Extract and surface the promotion reason clearly
  if command -v python3 &>/dev/null; then
    echo "$HEALTH" | python3 -c "
import sys, json
try:
    h = json.load(sys.stdin)
    print()
    print('   ┌─ Production model ──────────────────────────────────')
    print(f'   │  Type    : {h.get(\"model_type\", \"?\")}')
    print(f'   │  Version : v{h.get(\"version\", \"?\")}')
    print(f'   │  Test R² : {h.get(\"test_r2\", \"?\")}')
    print(f'   │  CV R²   : {h.get(\"cv_r2\", \"?\")}')
    print(f'   │  RMSE    : {h.get(\"test_rmse\", \"?\")}')
    print(f'   │  Reason  : {h.get(\"promotion_reason\", \"?\")}')
    print('   └─────────────────────────────────────────────────────')
except: pass
" 2>/dev/null || true
  fi
else
  echo "   (curl not available — check http://127.0.0.1:8000/health manually)"
fi

# ---------------------------------------------------------------------------
# Step 10: Analysis scripts — now run AFTER promotion so they query the
#          correct Production run (Ridge, not Linear Regression)
# ---------------------------------------------------------------------------
echo ""
echo "10. Running analysis scripts against Production model..."

echo "    10a. Coefficient analysis..."
python src/analyze_coefficients.py

echo "    10b. Error visualisation..."
python src/visualize_errors.py

echo "    10c. Multicollinearity analysis..."
python src/multicollinearity_analysis.py

echo "    10d. Outlier detective..."
python src/outlier_detective.py

echo "    ✅ All analysis scripts complete"

# ---------------------------------------------------------------------------
# Step 11: Generate dashboard HTML
# ---------------------------------------------------------------------------
echo ""
echo "11. Generating dashboard..."
python src/generate_dashboard.py

# ---------------------------------------------------------------------------
# Step 12: Open dashboard in browser
# ---------------------------------------------------------------------------
echo ""
echo "12. Opening dashboard..."
if command -v open &>/dev/null; then
  open dashboard.html          # macOS
elif command -v xdg-open &>/dev/null; then
  xdg-open dashboard.html      # Linux
else
  echo "    Open dashboard.html manually in your browser"
fi

# ---------------------------------------------------------------------------
# All systems running
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════╗"
echo "║   All systems running!               ║"
echo "╠══════════════════════════════════════╣"
echo "║  MLflow UI  → http://127.0.0.1:5000  ║"
echo "║  API server → http://127.0.0.1:8000  ║"
echo "║  API docs   → http://127.0.0.1:8000/docs ║"
echo "║  Dashboard  → dashboard.html         ║"
echo "╠══════════════════════════════════════╣"
echo "║  Logs:                               ║"
echo "║    logs/mlflow.log                   ║"
echo "║    logs/api.log                      ║"
echo "╠══════════════════════════════════════╣"
echo "║  Press Ctrl+C to stop everything.    ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "   Servers running. Press Ctrl+C to stop."
wait $MLFLOW_PID $API_PID
