# Create a dictionary to hold the parsed color data
color_dict = {}

def duplicate_characters(input_str):
    return ''.join([char * 2 for char in input_str])

# Open the txt file for reading
with open("color_map.txt", "r") as f:
    # Read lines one by one
    for line in f:
        # Remove any leading and trailing whitespace
        line = line.strip()

        # Split line by ':' to get the color name and values
        try:
            color_name, color_values = line.split(":")
        except ValueError:
            print(f"Skipping invalid line: {line}")
            continue

        # Remove quotes and whitespace from the color_name
        color_name = color_name.replace("\"", "").strip()

        # Remove brackets and whitespace from color_values, then split by comma
        color_values = color_values.replace("[", "").replace("]", "").strip()
        color_values = color_values.split(", ")
        print(color_values)
        color_values = [s.strip() for s in color_values]
        for color_hex in color_values:
            c_hex = duplicate_characters(color_hex)
            color_dict[c_hex] = color_name
        # Add to dictionary
        # color_dict[color_name] = color_values

# Print the parsed color dictionary
print(color_dict)

import json
with open("./hex_map.json", 'w') as f:
    json.dump(color_dict, f,indent=4)