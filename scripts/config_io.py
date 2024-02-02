import json
import os

CONFIG_PATH = os.path.join("configs", "config.json")

def read_config_value(key):

	# Check if file exists
	if not os.path.exists(CONFIG_PATH):
		return None, 1  # File does not exist

	try:
		with open(CONFIG_PATH, "r") as file:
			config = json.load(file)

		if key not in config:
			return None, 2  # Key not found

		if config[key] is None:
			return None, 3  # Key found but value is None

		return config[key], 0  # Success

	except Exception as e:
		print(f"An error occurred: {e}")
		return None, 4  # Other errors
