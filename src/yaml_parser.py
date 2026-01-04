import yaml


class YamlParser(object):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    def __getitem__(self, key):
        return self.config[key]

    def get_flat_config(self):
        """Returns a flattened version of the config for MLflow logging."""
        def flatten(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        return flatten(self.config)