import yaml

# Load YAML file
def load_yaml(config_path: str):
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config