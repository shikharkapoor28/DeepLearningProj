import itertools
import yaml
from typing import List, Dict

class AblationRunner:
    """
    Automates running multiple variations of the environment/agent 
    to determine the contribution of individual components.
    """
    def __init__(self, base_config_path: str):
        with open(base_config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

    def generate_sweep_configs(self, sweep_params: Dict[str, List]) -> List[Dict]:
        """
        Creates a list of configuration dictionaries for every combination of parameters.
        Example sweep_params:
        {
            "model_architecture.encoder": ["transformer", "mlp"],
            "environment.trading_fee_bps": [0, 6]
        }
        """
        keys = list(sweep_params.keys())
        values = list(sweep_params.values())
        combinations = list(itertools.product(*values))
        
        configs = []
        for combo in combinations:
            config_copy = self._deep_copy(self.base_config)
            for key, val in zip(keys, combo):
                self._set_nested_val(config_copy, key, val)
            configs.append(config_copy)
            
        return configs

    def run_ablation_study(self, sweep_configs: List[Dict]):
        """
        Executes a training run for each configuration.
        """
        results = []
        for i, config in enumerate(sweep_configs):
            print(f"Running ablation {i+1}/{len(sweep_configs)}")
            # Initialize Trainer with new config
            # trainer.train()
            # metrics = evaluator.run_backtest()
            # results.append((config, metrics))
            pass
        return results

    def _deep_copy(self, config):
        # Quick robust deep copy using YAML
        return yaml.safe_load(yaml.dump(config))
        
    def _set_nested_val(self, d: dict, nested_key: str, value):
        keys = nested_key.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
