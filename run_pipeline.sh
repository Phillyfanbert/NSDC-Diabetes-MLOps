#!/bin/bash
# run_pipeline.sh — full MLOps pipeline + serving in one command
# ==============================================================
# Usage: bash run_pipeline.sh
#
# Runs the 5-step data/training pipeline, then launches:
#   - MLflow UI        → http://127.0.0.1:5000
#   - FastAPI server   → http://127.0.0.1:8000
#   - Dashboard        → dashboard.html (opens in browser)
#
# Press Ctrl+C once to shut everything down cleanly.

set -e

# ---------------------------------------------------------------------------
# Cleanup: kill background servers when the script exits (Ctrl+C or error)
# ---------------------------------------------------------------------------
cleanup() {
  echo ""
  echo "Shutting down servers..."
  kill "$MLFLOW_PID" "$API_PID" 2>/dev/null
  echo "Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 1–5: Data pipeline
# ---------------------------------------------------------------------------
echo "==============================="
echo " NSDC Diabetes MLOps Pipeline"
echo "==============================="

echo ""
echo "1. Fetching data..."
python src/fetch_data.py

echo ""
echo "2. Validating data..."
python src/validate_data.py

echo ""
echo "3. Cleaning data..."
python src/cleaning.py

echo ""
echo "4. Engineering features..."
python src/features.py

echo ""
echo "5. Training model & logging to MLflow..."
python src/train_model.py

# ---------------------------------------------------------------------------
# Step 6: Generate dashboard
# ---------------------------------------------------------------------------
echo ""
echo "6. Generating dashboard..."
python src/generate_dashboard.py

# ---------------------------------------------------------------------------
# Step 7: Start MLflow UI in background
# ---------------------------------------------------------------------------
echo ""
echo "7. Starting MLflow UI at http://127.0.0.1:5000 ..."
mlflow ui --host 127.0.0.1 --port 5000 > logs/mlflow.log 2>&1 &
MLFLOW_PID=$!

# Give MLflow a moment to start before the API tries to connect
echo "   Waiting for MLflow to be ready..."
sleep 4

# ---------------------------------------------------------------------------
# Step 8: Start FastAPI server in background
# ---------------------------------------------------------------------------
echo ""
echo "8. Starting API server at http://127.0.0.1:8000 ..."
mkdir -p logs
python src/serve_model.py > logs/api.log 2>&1 &
API_PID=$!

# Give the API a moment to load the model
echo "   Waiting for API to be ready..."
sleep 5

# ---------------------------------------------------------------------------
# Step 9: Open dashboard in browser
# ---------------------------------------------------------------------------
echo ""
echo "9. Opening dashboard..."
if command -v open &>/dev/null; then
  open dashboard.html          # macOS
elif command -v xdg-open &>/dev/null; then
  xdg-open dashboard.html      # Linux
fi

# ---------------------------------------------------------------------------
# Done — show status and keep running until Ctrl+C
# ---------------------------------------------------------------------------
echo ""
echo "==============================="
echo " All systems running!"
echo "==============================="
echo " MLflow UI  → http://127.0.0.1:5000"
echo " API server → http://127.0.0.1:8000"
echo " API docs   → http://127.0.0.1:8000/docs"
echo " Dashboard  → dashboard.html"
echo ""
echo " Logs: logs/mlflow.log | logs/api.log"
echo ""
echo " Press Ctrl+C to stop everything."
echo "==============================="

# Keep script alive so trap fires on Ctrl+C
wait