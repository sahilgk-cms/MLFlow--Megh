import yaml


def load_search_space_config(search_space_path: str,
                             model_name: str, optimizer_type: str) -> dict: 
    with open(search_space_path, "r") as f:
        config = yaml.safe_load(f)

    return config[model_name][optimizer_type]


def build_optuna_search_space(trial, space_config: dict):
    params = {}

    for param, spec in space_config.items():
        if spec["type"] == "int":
            params[param] = trial.suggest_int(
                param,
                spec["low"],
                spec["high"],
                step = spec.get("step", 1)
            )
        elif spec["type"] == "float":
            params[param] = trial.suggest_float(
                param,
                spec["low"],
                spec["high"],
                log = spec.get("log", False)
            )
    return params



def get_search_space(search_space_path: str, 
                     model_name: str, optimizer_type: str):
 
    space_config = load_search_space_config(search_space_path, model_name, optimizer_type)

    if optimizer_type == "optuna":

        # return function that Optuna will call with trial
        def param_space(trial):
            return build_optuna_search_space(trial, space_config)

        return param_space

    else:
        return space_config

