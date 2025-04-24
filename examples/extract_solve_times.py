import re
import pathlib
from collections import deque
from typing import List, Optional, Tuple

def extract_solve_times(log_file_path: str) -> List[float]:
    """
    Extracts solve times for step 1 from a log file based on contact mode.

    Args:
        log_file_path: Path to the log file.

    Returns:
        A list of solve times (float) where the contact mode two lines
        prior was 'index' or 'thumb_middle'.
    """
    solve_times: List[float] = []
    # Regex to match the contact mode line (e.g., "14 index" or "16 thumb_middle")
    contact_mode_regex = re.compile(r'^\d+ (index|thumb_middle)$')
    # Regex to extract the time from the "Solve time" line
    solve_time_regex = re.compile(r'Solve time for step 1.*?([\d.]+)$')

    try:
        with open(log_file_path, 'r') as f:
            # Use deque to efficiently keep track of the last 3 lines
            lines_buffer = deque(maxlen=5)
            for line in f:
                lines_buffer.append(line.strip())
                if len(lines_buffer) == 5:
                    current_line = lines_buffer[4]
                    line_minus_2 = lines_buffer[0]

                    # Check if the current line is a "Solve time for step 1" line
                    solve_time_match = solve_time_regex.match(current_line)
                    # print(solve_time_match)
                    if solve_time_match:
                        # Check if the line two positions before matches the contact mode criteria
                        contact_mode_match = contact_mode_regex.match(line_minus_2)
                        # print(line_minus_2)
                        if contact_mode_match:
                            # Extract the time and add to the list
                            time_str = solve_time_match.group(1)
                            try:
                                solve_times.append(float(time_str))
                            except ValueError:
                                print(f"Warning: Could not convert time '{time_str}' to float in line: {current_line}")

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return solve_times

# --- Main execution ---
if __name__ == "__main__":
    LOG_FILE = "/home/abhinav/Documents/ccai/examples/logs/corl_screwdriver/allegro_screwdriver_recovery_model_mlp_ablation_fixed_rand_pct_pi_2_damping_.1_eval.log"
    OUTPUT_FILE = "/home/abhinav/Documents/ccai/extracted_solve_times_list.py"

    extracted_times = extract_solve_times(LOG_FILE)
    import numpy as np
    print(np.mean(extracted_times))

    # Write the extracted times to the output file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write("# filepath: /home/abhinav/Documents/ccai/extracted_solve_times_list.py\n")
            f.write("# Extracted solve times for step 1 where contact mode was 'index' or 'thumb_middle'\n")
            f.write("solve_times = [\n")
            for time in extracted_times:
                f.write(f"    {time},\n")
            f.write("]\n")
        print(f"Successfully extracted {len(extracted_times)} solve times to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing to output file {OUTPUT_FILE}: {e}")

