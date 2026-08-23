import json, sys, os, io, warnings
from contextlib import redirect_stdout, redirect_stderr

# Ensure working directory is workspace root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

notebook_path = "work/notebooks/w07_action_playbook.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution context
exec_globals = {"__name__": "__main__"}

# Custom display function
def custom_display(obj):
    if hasattr(obj, 'to_string'):
        print(obj.to_string())
    elif hasattr(obj, '_repr_html_'):
        print(obj.to_string() if hasattr(obj, 'to_string') else str(obj))
    else:
        print(obj)

exec_globals['display'] = custom_display

print(f"Executing {notebook_path} top to bottom...")

cell_execution_count = 1
for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        print(f"\n--- Executing Code Cell {cell_execution_count} ---")
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)
            
            out_text = stdout_capture.getvalue()
            err_text = stderr_capture.getvalue()
            
            outputs = []
            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(True)
                })
            if err_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": err_text.splitlines(True)
                })
                
            cell["outputs"] = outputs
            cell["execution_count"] = cell_execution_count
            print(f"Cell {cell_execution_count} executed successfully.")
            if out_text:
                print("Output summary:")
                print(out_text[:500])
                
        except Exception as e:
            print(f"ERROR in cell {cell_execution_count}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        cell_execution_count += 1

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nNotebook executed top to bottom and saved successfully!")
