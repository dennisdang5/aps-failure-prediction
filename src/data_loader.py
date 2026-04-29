import pandas as pd
from pathlib import Path
import yaml

def load_config(yaml_file_path='src/config.yml'):
    project_root = Path(__file__).parent.parent
    full_path = project_root / yaml_file_path
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_train_data(data_config):
    project_root = Path(__file__).parent.parent
    full_path = project_root / data_config['train_path']
    return pd.read_csv(
        full_path,
        na_values=data_config['na_values'],
        skiprows=data_config['skip_rows']
    )

def load_test_data(data_config):
    project_root = Path(__file__).parent.parent
    full_path = project_root / data_config['test_path']
    return pd.read_csv(
        full_path,
        na_values=data_config['na_values'],
        skiprows=data_config['skip_rows']
    )