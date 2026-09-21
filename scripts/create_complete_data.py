# Generate complete 2000 CDR dataset ensuring all suspects, cell towers, and interactions are represented accurately
import pandas as pd
import numpy as np

# Load suspects and edges
suspects_df = pd.read_csv("data/suspects.csv")
edges_df = pd.read_csv("data/network_edges.csv")

suspect_dict = dict(zip(suspects_df['suspect_id'], suspects_df['phone_number']))
suspect_ids = list(suspect_dict.keys())
cities = ["Delhi", "Mumbai", "Pune", "Jaipur", "Ahmedabad", "Surat", "Nagpur", "Indore", "Bhopal", "Lucknow", "Kanpur", "Patna", "Ranchi", "Amritsar", "Chandigarh"]
call_types = ["Voice", "Voice", "Voice", "SMS", "SMS", "Data"]

# Read existing CDR rows if any
existing_rows = []
try:
    existing_df = pd.read_csv("data/cdr_records.csv")
    existing_rows = existing_df.values.tolist()
except Exception:
    existing_rows = []

print(f"Existing CDR rows: {len(existing_rows)}")

# If we need 2000 total rows:
rng = np.random.RandomState(42)

all_rows = existing_rows[:]
current_count = len(all_rows)

edge_pairs = edges_df[['source_suspect_id', 'target_suspect_id']].values.tolist()

for i in range(current_count + 1, 2001):
    call_id = f"CDR{i:05d}"
    # 70% chance to pick connected edge, 30% random
    if rng.rand() < 0.7 and len(edge_pairs) > 0:
        pair_idx = rng.randint(0, len(edge_pairs))
        caller_id, callee_id = edge_pairs[pair_idx]
        if rng.rand() > 0.5:
            caller_id, callee_id = callee_id, caller_id
    else:
        caller_id = rng.choice(suspect_ids)
        callee_id = rng.choice(suspect_ids)
        while callee_id == caller_id:
            callee_id = rng.choice(suspect_ids)
            
    caller_num = suspect_dict.get(caller_id, "+91-9800000000")
    callee_num = suspect_dict.get(callee_id, "+91-9800000001")
    
    # Random date in 2026
    month = rng.randint(1, 9)
    day = rng.randint(1, 29)
    hour = rng.randint(0, 24)
    minute = rng.randint(0, 60)
    second = rng.choice([0, 15, 30, 45])
    timestamp = f"2026-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    
    duration = int(rng.randint(10, 1800))
    call_type = rng.choice(call_types)
    city = rng.choice(cities)
    
    all_rows.append([call_id, caller_id, caller_num, callee_id, callee_num, timestamp, duration, call_type, city])

cols = ["call_id", "caller_id", "caller_number", "callee_id", "callee_number", "timestamp", "duration_seconds", "call_type", "cell_tower_city"]
cdr_df = pd.DataFrame(all_rows[:2000], columns=cols)
cdr_df.to_csv("data/cdr_records.csv", index=False)
print(f"data/cdr_records.csv successfully created with {len(cdr_df)} rows.")
