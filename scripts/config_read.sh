#!/bin/bash

# Exit if no key was provided
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

# Use awk to extract the value for the key
VALUE=$(grep -o "\"$KEY\":.*" $CONFIG_FILE | head -n 1 | sed -E 's/.*: *"?([^",]+)"?,?/\1/' | tr -d '"' | tr -d '\r')

# Handle different cases based on the value extracted
if [ -z "$VALUE" ]; then
	echo "Key not found."
	exit 2
elif [ "$VALUE" == "null" ]; then
	echo "Key found but value is null."
	exit 3
else
	# Output the value and exit successfully
	echo "$VALUE"
	exit 0
fi
