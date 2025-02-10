import os

# Set the directory you want to work in (use "." for current directory)
folder_path = "/home/adamhung/code/ccai/data/regrasp_to_turn_datasets"

# Define the old and new prefixes
old_prefix = "regrasp_to_turn_dataset_narrow_plan"
new_prefix = "regrasp_to_turn_dataset_narrow_x"

# Loop through all items in the folder
for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    # Check if it's a file and starts with the specified prefix
    if os.path.isfile(full_path) and filename.startswith(old_prefix):
        # Keep the part of the filename after the old prefix
        suffix = filename[len(old_prefix):]
        new_filename = new_prefix + suffix
        new_full_path = os.path.join(folder_path, new_filename)
        print(f"Renaming '{filename}' to '{new_filename}'")
        os.rename(full_path, new_full_path)