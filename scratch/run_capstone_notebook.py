import json
import sys
from pathlib import Path

nb_path = Path("work/notebooks/capstone.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

print(f"Executing {nb_path} with {len(nb['cells'])} cells...", flush=True)

global_env = {}

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"--- Running Code Cell {idx+1} ---", flush=True)
        try:
            exec(source, global_env)
            print(f"Cell {idx+1} executed successfully.", flush=True)
        except Exception as e:
            print(f"Error executing cell {idx+1}: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

print("=== ALL CAPSTONE NOTEBOOK CELLS EXECUTED SUCCESSFULLY TOP-TO-BOTTOM ===", flush=True)
