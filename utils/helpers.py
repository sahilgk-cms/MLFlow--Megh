import json
import importlib
import yaml

def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def safe_tag_value(value):
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)

def load_config(module_path, var_name):
    module = importlib.import_module(module_path)
    return getattr(module, var_name)