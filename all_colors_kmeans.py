import numpy as np
from sklearn.cluster import KMeans
import json

# RGB values from (0, 0, 0) to (255, 255, 255)
rgb_values = np.array(
    [[r, g, b] for r in range(256) for g in range(256) for b in range(256)]
)

num_clusters = 40

# Kmeans
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(rgb_values)

# Create a dict to store clusters and their associated RGB values
clustered_colors = {i: [] for i in range(num_clusters)}

dic = {}
for i, label in enumerate(cluster_labels):
    clustered_colors[label].append(tuple(rgb_values[i]))
    dic[str(i)] = kmeans.cluster_centers_[i]
    

colors = kmeans.cluster_centers_
print(colors)
with open("./kmeans_map_40.json",'w') as j:
    json.dump(dic, j, indent=4)

