# config.py
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

config = load_config()
