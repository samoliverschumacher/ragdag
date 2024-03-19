import json
import os
from pathlib import Path

import dotenv
import yaml

from app import ROOT_DIR

dotenv.load_dotenv()

QDRANT_URL = 'http://localhost:6333'
REDACTED_INFORMATION_TOKEN_MAP = { 'Persons names': '<REDACTED: Persons names>',
                                   'Financial information': '<REDACTED: Financial information>' }


def load_config(file_path: str | Path) -> dict | None:
    """PLACEHOLDER for a config store service.

    Load the config from the given file .json or .yaml path
    """
    _, file_extension = os.path.splitext(file_path)
    try:
        with open(file_path) as f:
            if file_extension == '.json':
                return json.load(f)
            elif file_extension in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError('Unsupported file type')
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_config(config_name: str) -> dict:
    """Get the config for the given name"""
    return _get_config(config_name)


def _get_config(config_name: str) -> dict:
    """Get the config for the given name. If the config is not found, raise an error"""

    key = os.environ.get("CONFIG_STORE_KEY")
    if not key:
        print("CONFIG_STORE_KEY not set. This would throw an error in production")

    config_name = config_name.replace('-', '_') + '_config'
    if config_name not in config:
        raise ValueError(f'Unknown config name: {config_name}')

    return config[config_name]


# In production this would be a config store accesible using a key from .env
fpath = os.environ.get('CONFIG', ROOT_DIR.parent / 'config.yaml')
config = load_config(fpath)
