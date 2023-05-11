import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

start_time = time.time()

# Generate some random data
X = np.random.rand(100000, 2)

# Define the number of clusters
num_clusters = 10

# Initialize the KMeans object with the number of clusters
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the data
kmeans.fit(X)

# Get the centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

end_time = time.time()

# Print the centroids and labels
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

# fig, ax = plt.subplots()
# ax.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=150, c='black')
# for i, label in enumerate(labels):
#     ax.annotate(label, (centroids[label] + 0.01))
# # Scatter plot the centroids
# plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids')

# Scatter plot the data points with their corresponding labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

duration = end_time - start_time

print("Execution time:", duration, "seconds")
