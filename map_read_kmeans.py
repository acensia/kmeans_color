# Create a dictionary to hold the parsed color data
color_dict = {}

def duplicate_characters(input_str):
    return ''.join([char * 2 for char in input_str])

kmeans_dict = {}
# Open the txt file for reading
with open("color_byKmeans.txt", "r") as f:
    # Read lines one by one
    for line in f:
        # Remove any leading and trailing whitespace
        line = line.strip()

        # Split line by ':' to get the color name and values
        colors = line.split(",")
        for i, c in enumerate(colors):
            c = c.replace("[", "").replace("]", "").strip()
            r, g, b = c.split()
            rgb = [r, g, b]
            kmeans_dict[i] = rgb
        
import json
with open("./kmeans_map.json", 'w') as f:
    json.dump(kmeans_dict, f,indent=4)