#This file is responsible for loading, cleaning, and preprocessing the CIC-IDS dataset. It uses memory-optimized techniques, handles missing/infinite values, applies robust scaling, and partitions the dataset for federated simulation.
import os
import numpy as np
import pandas as pd
from dask import dataframe as dd
from sklearn.preprocessing import RobustScaler
# ====== File Path Validation ======
file_path = r"C:\Users\dhruv\Desktop\project\Fedrated_Privacy_Proj\02-14-2018.csv"
assert os.path.exists(file_path), f"File not found at {file_path}"

# ====== Memory-Optimized Loading ======
dtypes = {
    'Flow Duration': 'uint32',
    'Tot Fwd Pkts': 'uint16',
    'Flow Byts/s': 'float32',
    'Flow Pkts/s': 'float32',
    'Label': 'category'
}

# Load in chunks if memory constrained
def process_chunk(chunk):
    return chunk.replace([np.inf, -np.inf], np.nan)

df = pd.read_csv(file_path, dtype=dtypes, low_memory=False, 
                 parse_dates=['Timestamp'], 
                 infer_datetime_format=True)

# ====== Critical Column Checks ======
print("Initial Data Shape:", df.shape)
print("Missing Values:\n", df.isna().sum())
print("Label Categories:", df['Label'].unique())

# ====== Infinite Value Handling ======
inf_cols = ['Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Max', 'Idle Max']
df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)

# Protocol-aware imputation
for col in inf_cols:
    df[col] = df.groupby('Protocol', observed=True)[col].transform(
        lambda x: x.fillna(x.median())
    )

# ====== Irrelevant Column Removal ======
cols_to_drop = [
    'Timestamp', 'Fwd URG Flags', 'Bwd URG Flags', 
    'Init Fwd Win Byts', 'Init Bwd Win Byts'
]
df = df.drop(columns=cols_to_drop)

# ====== Categorical Conversion ======
df['Protocol'] = df['Protocol'].astype('category').cat.codes  # TCP=0, UDP=1
df['Dst Port'] = df['Dst Port'].astype('category')

# ====== Robust Scaling ======
scaler = RobustScaler(quantile_range=(5, 95), 
                      with_centering=False,  # Avoid negative values
                      unit_variance=True)

robust_features = [
    'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Max', 'Idle Max'
]

# Ensure float32 to prevent overflow
df[robust_features] = df[robust_features].astype('float32')

# Quantile-based clipping (prevent post-scaling outliers)
for col in robust_features:
    q1 = df[col].quantile(0.05)
    q3 = df[col].quantile(0.95)
    df[col] = np.clip(df[col], q1, q3)

# Apply scaling
df[robust_features] = scaler.fit_transform(df[robust_features])
# ====== Federated Client Simulation ======
# Strategy 1: Split by protocol type
tcp_data = df[df['Protocol'] == 0].sample(frac=0.5, random_state=42)
udp_data = df[df['Protocol'] == 1].sample(frac=0.5, random_state=42)

# Strategy 2: Temporal splitting (using original timestamp order)
df_sorted = df.sort_values('Flow Duration')
client_count = 5
client_datasets = np.array_split(df_sorted, client_count)

# Save partitions
for i, client_df in enumerate(client_datasets):
    client_df.to_parquet(
        f'client_{i}.parquet',
        engine='pyarrow',
        compression='ZSTD'
    )
# ====== Post-Processing Verification ======
assert not df[robust_features].isnull().any().any(), "NaNs present!"
assert not np.isinf(df[robust_features]).any().any(), "Infinite values!"
assert df[robust_features].max().max() < 100, "Scaling overflow"
assert df[robust_features].min().min() >= 0, "Negative scaled values"

# Label distribution check
label_dist = df['Label'].value_counts(normalize=True)
assert label_dist.min() > 0.01, "Severe class imbalance remains"

# Memory check (target <4GB)
print("Final Memory Usage:", df.memory_usage().sum()/1024**3, "GB")

# Save processed dataset
df.to_parquet('processed_data.parquet', 
             engine='pyarrow',
             compression='ZSTD',
             index=False)

# Save scaler for federated clients
import joblib
joblib.dump(scaler, 'robust_scaler.pkl')

