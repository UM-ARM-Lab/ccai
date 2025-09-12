#!/bin/bash

# Set the checkpoint path
# CORL
# data_dirs=(
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_50.pt" # ndf seed 1
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_51.pt" 
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_52.pt" 
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_53.pt" 
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_54.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_100.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_101.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_102.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_103.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-16_15-30-22/nn/state_data_104.pt"      
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_50.pt" # confid seed 0
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_51.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_52.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_53.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_54.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_100.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_101.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_102.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_103.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-14_16-47-57/state_data_104.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_50.pt" # pointcloud seed 0
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_51.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_52.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_53.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_54.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_100.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_101.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_102.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_103.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-04-15_15-47-27/state_data_104.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_50.pt" # dexpoint seed 1
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_51.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_52.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_53.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_54.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_100.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_101.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_102.pt"
#     "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_103.pt"
#     # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-05-01_18-40-22/state_data_104.pt"
# )

# ICRA
#!/bin/bash

# Set the checkpoint path
# Define run base directories (the directory that directly contains the checkpoint files).
# If a run stores checkpoints in an 'nn' subfolder, include it here.
runs=(
    "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-09-10_22-42-31/nn"         # ndf seed 1 (has nn/)
    # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-09-08_14-37-35/nn"           # confid seed 0
    # "/home/fanyang/github/DynamicCatching/catching/logs/rl_games/allegro_screwdriver/2025-09-08_14-31-28/nn"           # dexpoint seed 0
)

# Define the common list of checkpoint IDs (used for every run)
ids=(105 106 107 108 109 100 101 102 103 104)
cuda_id=1
# Build the full list of checkpoint files
data_dirs=()
for run in "${runs[@]}"; do
    for id in "${ids[@]}"; do
        data_dirs+=( "${run}/state_data_${id}.pt" )
    done
done

# Loop through each directory and run the command sequentially
for dir in "${data_dirs[@]}"; do
    echo "Running eval_catching.py on $dir"
    CUDA_VISIBLE_DEVICES=${cuda_id} python3 eval_catching.py --data_dir="$dir"
done

echo "All evaluations completed."
