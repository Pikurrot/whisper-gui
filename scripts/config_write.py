import json
import sys
import os

def main(key, value):
	config_file = os.path.join("configs", "config.json")

	# Check if the config.json file exists
	if not os.path.exists(config_file):
		sys.exit(1)

	# Read the existing data
	try:
		with open(config_file, 'r') as file:
			try:
				data = json.load(file)
			except json.JSONDecodeError:
				# File is empty, create a new dictionary
				data = {}
	except IOError as e:
		sys.exit(1)

	value = str(value).lower()
	if value == "true":
		value = True
	elif value == "false":
		value = False
	elif value == "null":
		value = None
	data[key] = value

	# Write the updated data
	try:
		with open(config_file, 'w') as file:
			json.dump(data, file, indent=4)
	except IOError as e:
		print(f"Error writing to file: {e}")
		sys.exit(1)

	sys.exit(0)

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit(1)

	input_key = sys.argv[1]
	input_value = sys.argv[2]

	main(input_key, input_value)
