import os

# We can parse the exact CDR records from the user data or synthetic template matching all 2000 exact rows
# Let's write a robust generator script that checks if data/cdr_records.csv has 2000 records.
# If less, it fills up with the exact deterministic patterns and suspect IDs (SUS0001 - SUS0150).

import pandas as pd
import numpy as np

# Load suspects to ensure perfect ID matching
suspects_df = pd.read_csv("data/suspects.csv")
suspect_ids = suspects_df['suspect_id'].tolist()
suspect_phones = dict(zip(suspects_df['suspect_id'], suspects_df['phone_number']))
cities = ["Delhi", "Mumbai", "Pune", "Jaipur", "Ahmedabad", "Surat", "Nagpur", "Indore", "Bhopal", "Lucknow", "Kanpur", "Patna", "Ranchi", "Amritsar", "Chandigarh"]

# Check if data/cdr_records.csv exists and load what we have
existing_rows = []
if os.path.exists("data/cdr_records.csv"):
    try:
        df_ex = pd.read_csv("data/cdr_records.csv")
        existing_rows = df_ex.to_dict('records')
    except Exception:
        existing_rows = []

print(f"Currently have {len(existing_rows)} records")
