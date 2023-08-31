import json
import cv2
import numpy as np

with open("./kmeans_map.json", 'r') as j:
    k_dict = json.load(j)

# Create a new image
width, height = 200, 200

img = []
for i in range(3):
    img_h = []
    for j in range(5):
        paper = np.full((width, height, 3), k_dict[str(i*5 + j)], dtype=np.float32)
        img_h.append(paper)
    img_h_stacked = np.hstack(img_h)
    img.append(img_h_stacked)

img_stacked = np.vstack(img)
print(img_stacked.shape)
img_merged = cv2.cvtColor(img_stacked, cv2.COLOR_RGB2BGR)

cv2.imwrite("./kmeans_colos.jpg", img_merged)