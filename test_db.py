import sqlite3
import os

db_path = os.path.join(os.getcwd(), 'kb_data', 'metadata.db')
print(f"Connecting to {db_path}...")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT run_id, status, progress, length(result_json), error FROM runs ORDER BY created_at DESC LIMIT 1;")
    row = cursor.fetchone()
    if row:
        run_id, status, progress, res_len, error = row
        print(f"Recent Run -> ID: {run_id}")
        print(f"Status: {status}")
        print(f"Progress: {progress}")
        print(f"Result JSON Length: {res_len}")
        print(f"Error: {error}")
    else:
        print("No runs found in database.")
    
    conn.close()
except Exception as e:
    import traceback
    traceback.print_exc()
