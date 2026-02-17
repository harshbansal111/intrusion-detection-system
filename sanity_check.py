import json
import glob
import sys

def check_notebooks():
    notebooks = glob.glob("**/*.ipynb", recursive=True)
    failed = False
    for nb_path in notebooks:
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                json.load(f)
            print(f"OK: {nb_path}")
        except json.JSONDecodeError as e:
            print(f"FAIL: {nb_path} - {e}")
            failed = True
        except Exception as e:
            print(f"FAIL: {nb_path} - {e}")
            failed = True
    
    if failed:
        sys.exit(1)
    else:
        print("All notebooks are valid JSON.")
        sys.exit(0)

if __name__ == "__main__":
    check_notebooks()
