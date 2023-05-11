from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

start_time = time.time()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate some random data
X = None
if rank == 0:
    X = np.random.rand(60000, 2)

# Scatter the data to all the processors
local_X = np.zeros((X.shape[0] // size, X.shape[1]))
comm.Scatter(X, local_X, root=0)

# Define the number of clusters
num_clusters = 10

# Initialize the KMeans object with the number of clusters
kmeans = KMeans(n_clusters=num_clusters)

# Fit the model to the local data
kmeans.fit(local_X)

# Get the local centroids and labels
local_centroids = kmeans.cluster_centers_
local_labels = kmeans.labels_

# Gather the local centroids and labels to the root process
centroids = None
labels = None
if rank == 0:
    centroids = np.zeros((num_clusters, X.shape[1]))
    labels = np.zeros(X.shape[0], dtype=np.int)
centroids_counts = np.zeros(num_clusters, dtype=np.int)
comm.Reduce(local_centroids, centroids, op=MPI.SUM, root=0)
comm.Reduce(local_labels, labels, op=MPI.SUM, root=0)
comm.Reduce(np.array([len(local_X)]*num_clusters), centroids_counts, op=MPI.SUM, root=0)

# Normalize the centroids
if rank == 0:
    for i in range(num_clusters):
        if centroids_counts[i] > 0:
            centroids[i] /= centroids_counts[i]

# Broadcast the centroids to all the processors
comm.Bcast(centroids, root=0)

# Assign the data points to the nearest centroids
distances = np.linalg.norm(local_X[:, np.newaxis, :] - centroids, axis=2)
local_assignments = np.argmin(distances, axis=1)

# Gather the local assignments to the root process
assignments = None
if rank == 0:
    assignments = np.zeros(X.shape[0], dtype=np.int)
counts = np.zeros(num_clusters, dtype=np.int)
comm.Reduce(local_assignments, assignments, op=MPI.SUM, root=0)
comm.Reduce(np.array([len(local_X)]*num_clusters), counts, op=MPI.SUM, root=0)

end_time = time.time()

# Print the final assignments and centroids
if rank == 0:
    print("Final Assignments:")
    print(assignments)
    print("Final Centroids:")
    print(centroids)

# Scatter plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids')

# Scatter plot the data points with their corresponding labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label='Data Points')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

duration = end_time - start_time

print("Execution time:", duration, "seconds")
