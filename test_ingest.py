# test_ingest.py
from data_ingest import load_koi

df, feats = load_koi()

print("✅ Downloaded KOI cumulative sample")
print("Rows:", len(df))
print("Using feature columns:", feats)
print("\nLabel counts:")
print(df["koi_disposition"].value_counts())

print("\nPreview:")
print(df.head(5))