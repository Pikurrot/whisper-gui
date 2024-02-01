#!/bin/bash

# Check if a key was provided
if [ -z "$1" ]; then
	echo "No key provided."
	exit 1
fi

KEY="$1"
CONFIG_FILE="configs/config.json"

# Check if the config file exists and is readable
if [ ! -f "$CONFIG_FILE" ] || [ ! -r "$CONFIG_FILE" ]; then
	echo "Unable to read the configuration file."
	exit 1
fi

# Attempt to read the value for the key using grep and sed
VALUE_LINE=$(grep -o "\"$KEY\": *\"[^\"]*\"" "$CONFIG_FILE")

if [ -z "$VALUE_LINE" ]; then
	# Key not found in the file
	exit 2
fi

# Extract the value from the line
VALUE=$(echo "$VALUE_LINE" | sed -E "s/.*\"$KEY\": *\"([^\"]*)\".*/\1/")

if [ "$VALUE" == "null" ] || [ -z "$VALUE" ]; then
	# Key found but value is null or empty
	exit 3
fi

# Output the value and exit
echo "$VALUE"
exit 0
