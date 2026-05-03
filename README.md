# NSDC-Diabetes-MLOps
Here is a clean, professional "Getting Started" section you can copy and paste directly into your `README.md` file. It assumes your teammates have `conda` installed, which is standard for data science projects.

***

### Getting Started

This project uses an automated pipeline to process data and train machine learning models. Follow these steps to set up your environment and run the pipeline.

#### 1. Clone the Repository
```bash
git clone https://github.com/Phillyfanbert/NSDC-Diabetes-MLOps.git
cd NSDC-Diabetes-MLOps
```

#### 2. Create the Virtual Environment
We use `conda` to ensure all dependencies are isolated and consistent across machines:
```bash
conda create -n diabetes_mlops python=3.9 -y
conda activate diabetes_mlops
```

#### 3. Install Dependencies
Install the required libraries from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### 4. Run the Pipeline
We have automated the entire data-to-model workflow into a single script. Execute it with:
```bash
bash run_pipeline.sh
```

#### 5. Analyze Results
Once the pipeline finishes, launch the MLflow UI to view your experiment metrics (RMSE, R²), parameters, and model artifacts:
```bash
mlflow ui
```
*Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser to view the dashboard.*

***
 If you are running this for the first time on a new machine, `run_pipeline.sh` will handle all steps in order:
1. `fetch_data.py` (Downloads WHO data)
2. `validate_data.py` (Checks data quality)
3. `cleaning.py` (Standardizes formatting)
4. `features.py` (Engineers lags and scales features)
5. `train_model.py` (Trains the model and logs to MLflow)

***
