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

        if exit_code == 139:  # Segmentation fault exit code
            print("Segmentation fault detected. Restarting...")
            time.sleep(1)  # Optional delay before restarting
        else:
            print(f"Program exited with code: {exit_code}")
            break  # Exit the loop if it exits normally

CCAI_PATH = pathlib.Path(__file__).resolve().parents[2]
fpath = pathlib.Path(f'{CCAI_PATH}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart a Python script on crash.")
    parser.add_argument(
        "filename",
        type=str,
        help="The final filename of the Python script to be monitored and restarted."
    )
    args = parser.parse_args()
    script_path = fpath / "_value_function/test" / args.filename

    # Check if the constructed script path exists
    if not script_path.exists():
        print(f"Error: The script file '{script_path}' does not exist.")
        sys.exit(1)

    # Start monitoring and restarting on crash
    restart_on_crash(str(script_path))