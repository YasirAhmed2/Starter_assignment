import json
import os
import pandas as pd
import numpy as np
from io import StringIO
import sys

# Load notebook
nb_path = r"d:\FlyRank Internship\Starter_assignment\work\notebooks\w02_ml_task_framing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution scope
exec_globals = {}

exec_count = 1
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        cell["execution_count"] = exec_count
        cell["outputs"] = []
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            # We need to handle notebook magic or display if any, but our code is standard Python
            exec(code, exec_globals)
            out_str = redirected_output.getvalue()
            sys.stdout = old_stdout
            
            if out_str:
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_str.splitlines(keepends=True)
                })
        except Exception as e:
            out_str = redirected_output.getvalue()
            sys.stdout = old_stdout
            print(f"Error executing cell {exec_count}: {e}")
            cell["outputs"].append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": [f"{type(e).__name__}: {str(e)}"]
            })
            raise e
            
        exec_count += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook executed and saved successfully with all cell outputs captured.")
