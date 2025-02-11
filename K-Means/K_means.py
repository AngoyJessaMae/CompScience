def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    distance = sum((point1[i] - point2[i]) ** 2 for i in range(len(point1)))
    return distance ** 0.5

def initialize_centroids(data, k):
    """Manually initialize k centroids from the dataset."""
    return data[:k]

def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_index = distances.index(min(distances))
        clusters[closest_index].append(point)
    return clusters

def compute_new_centroids(clusters):
    """Compute new centroids as the mean of each cluster."""
    new_centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
            new_centroids.append(centroid)
    return new_centroids

def k_means(data, k, max_iters=100):
    """Perform K-Means clustering."""
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_new_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters


data = [
    [2, 5], [1, 8], [6, 2], [4, 3], [9, 1],
    [7, 5], [3, 6], [1, 7], [5, 8], [7, 6]
]

k = 3
centroids, clusters = k_means(data, k)
print("Final centroids:", centroids)
print("Clusters:", clusters)