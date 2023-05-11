import random
import matplotlib.pyplot as plt
from functools import reduce
import time

# Define a function to generate random points
def generate_random_points(num_points, max_coord):
    return [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(num_points)]

# Define a function to calculate the distance between two points
def distance(point1, point2):
    return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5

# Define the mapper function for MapReduce
def mapper(point, centroids):
    distances = [distance(point, centroid) for centroid in centroids]
    min_distance_index = distances.index(min(distances))
    return min_distance_index, (point, 1)

# Define the reducer function for MapReduce
def reducer(key, values):
    points, count = zip(*values)
    centroid = tuple(map(lambda x: sum(x) / len(x), zip(*points)))
    return centroid, sum(count)

# Define the K-means algorithm function
def k_means(points, k, num_iterations):
    # Initialize the centroids randomly
    centroids = random.sample(points, k)

    # Run the iterations of K-means algorithm
    for i in range(num_iterations):
        # MapReduce phase 1: Map
        mapped = [mapper(point, centroids) for point in points]
        
        # MapReduce phase 1: Shuffle and Sort (not necessary in this implementation)
        
        # MapReduce phase 2: Reduce
        reduced = {}
        for key, value in mapped:
            if key in reduced:
                reduced[key].append(value)
            else:
                reduced[key] = [value]
        
        # Update the centroids based on the reduced values
        centroids = [reducer(key, values)[0] for key, values in reduced.items()]

    # Return the final centroids and their assigned points
    assigned_points = {}
    for point in points:
        distances = [distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        if min_distance_index in assigned_points:
            assigned_points[min_distance_index].append(point)
        else:
            assigned_points[min_distance_index] = [point]

    return centroids, assigned_points

start_time = time.time()

# Generate some random points
points = generate_random_points(30000, 2)

# Run the K-means algorithm with 5 centroids and 10 iterations
centroids, assigned_points = k_means(points, 10, 5)

end_time = time.time()

print(centroids)
print(assigned_points)

# Plot the points and centroids on a graph
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'c', 'm']
for i, centroid in enumerate(centroids):
    color = colors[i % len(colors)]
    plt.scatter(centroid[0], centroid[1], s=100, c=color, marker='x')
    plt.scatter(*zip(*assigned_points[i]), s=50, c=color)
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

duration = end_time - start_time

print("Execution time:", duration, "seconds")
