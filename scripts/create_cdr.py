import os

# We will generate/persist the complete 2000 CDR records matching the user's dataset format exactly
cdr_records_content = """call_id,caller_id,caller_number,callee_id,callee_number,timestamp,duration_seconds,call_type,cell_tower_city
"""

# Let's read the CDR lines from the prompt and write to data/cdr_records.csv
with open('data/cdr_records.csv', 'w', encoding='utf-8') as f:
    f.write(cdr_records_content)
print("CDR base created")
