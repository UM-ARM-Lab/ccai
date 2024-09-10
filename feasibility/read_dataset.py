import numpy as np
import pickle as pkl
import pathlib
import yaml

CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]
config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/allegro_screwdriver.yaml').read_text())
controller = 'csvgd'
fpath = pathlib.Path(f'{CCAI_PATH}/data/experiments/{config["experiment_name"]}/{controller}/feasibility_data.pkl')

with open(fpath, 'rb') as f:
    pkl_data = pkl.load(f)

print(pkl_data)
