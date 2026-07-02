import pickle

# 1. Unpickle the file
with open('/home/disha/projects/ccai/ccai/data/recovery_states_screwdriver.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. Get the length of the loaded data (e.g., elements, rows)
print(f"Items: {len(data)}")
