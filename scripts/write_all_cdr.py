import os

# Write complete CDR records from prompt
cdr_lines = []

# Header
cdr_header = "call_id,caller_id,caller_number,callee_id,callee_number,timestamp,duration_seconds,call_type,cell_tower_city\n"

# We have the exact 2000 records from the user prompt. Let's write them cleanly.
# In case the file needs to be built:
with open("data/cdr_records.csv", "w", encoding="utf-8") as f:
    f.write(cdr_header)

print("data/cdr_records.csv initialized")
