import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import os

# Configuration: Constants at the top make the script easy to update
INDICATORS = {
    "target_diabetes": "NCD_GLUC_04", 
    "feature_obesity": "NCD_BMI_30A"
}
RAW_DATA_PATH = Path("data/raw/diabetes_obesity_raw.parquet")

def fetch_who_data(code):
    """Fetches raw JSON from WHO API and converts to DataFrame."""
    try:
        url = f"https://ghoapi.azureedge.net/api/{code}"
        print(f"📡 Requesting: {code}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status() # Raise error for bad status codes
        return pd.DataFrame(r.json()['value'])
    except Exception as e:
        print(f"❌ Error fetching {code}: {e}")
        return None

def clean_and_prepare(df, value_name):
    """Filters for both sexes and standardizes join keys."""
    if df is None or df.empty:
        return None
    
    # MLOps Lesson: Standardizing keys ensures the 'Data Contract' is met
    # Filter for Both Sexes (BTSX)
    df = df[df['Dim1'] == 'SEX_BTSX'].copy()
    
    # Standardize join keys to lowercase for consistency
    df = df.rename(columns={
        'SpatialDim': 'country_code', 
        'TimeDim': 'year', 
        'NumericValue': value_name
    })
    return df

def run_ingestion_pipeline():
    """Main execution logic for the data ingestion layer."""
    print("🚀 Starting Data Ingestion Pipeline...")
    
    # Ensure directory exists
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fetching
    data_frames = {}
    for label, code in INDICATORS.items():
        df = clean_and_prepare(fetch_who_data(code), label)
        if df is not None:
            data_frames[label] = df

    # 2. Merging
    if len(data_frames) == 2:
        # Join keys that exist in both datasets (Metadata columns)
        join_keys = ['country_code', 'year', 'Dim1', 'ParentLocation']
        
        print("🔗 Merging datasets and preserving metadata...")
        master_df = pd.merge(
            data_frames['target_diabetes'], 
            data_frames['feature_obesity'], 
            on=join_keys, 
            suffixes=('_diabetes', '_obesity')
        )

        # 3. Versioning/Storage
        # We save as Parquet to preserve the 46 columns' data types
        master_df.to_parquet(RAW_DATA_PATH, index=False, engine='pyarrow')
        
        print("\n" + "="*30)
        print(f"✅ PIPELINE SUCCESSFUL")
        print(f"Saved to: {RAW_DATA_PATH}")
        print(f"Observations: {len(master_df)}")
        print(f"Total Columns: {len(master_df.columns)}")
        print("="*30)
    else:
        print("❌ Pipeline failed: One or more datasets could not be retrieved.")

if __name__ == "__main__":
    run_ingestion_pipeline()