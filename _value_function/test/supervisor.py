import subprocess
import sys
import time
import pathlib
import argparse

def restart_on_crash(script_path):
    while True:
        print(f"Starting program: {script_path}...")
        # Run the script as a subprocess
        process = subprocess.run([sys.executable, script_path])
        exit_code = process.returncode

        # if exit_code == 139:  # Segmentation fault exit code
        if exit_code == 0:  
            print("Program exited normally.")
            break
        else:
            print("Program died. Restarting... exit code:", exit_code)
            time.sleep(1)  # Optional delay before restarting

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart a Python script on crash.")
    parser.add_argument(
        "filename",
        type=str,
        help="The Python script to be monitored and restarted."
    )
    args = parser.parse_args()

    if args.filename == "data":
        script_path = fpath / "_value_function/data_collect/get_regrasp_to_turning_dataset.py"
    elif args.filename == "test":
        script_path = fpath / "_value_function/test/test_method.py"
    elif args.filename == "test2":
        script_path = fpath / "_value_function/test/compare_dataset_sizes.py"
    elif args.filename == "sweep":
        script_path = fpath / "_value_function/test/regrasp_weight_sweep_safe.py"
    else:
        print(f"Error: Not a valid argument.")
        exit()

    # Check if the constructed script path exists
    if not script_path.exists():
        print(f"Error: Not a valid argument.")
        sys.exit(1)

    # Start monitoring and restarting on crash
    restart_on_crash(str(script_path))