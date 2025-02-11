def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def get_neighbors(training_data, test_instance, k):
    """Find the k nearest neighbors for a given test instance."""
    distances = []
    for train_instance in training_data:
        distance = euclidean_distance(train_instance[:-1], test_instance)
        distances.append((train_instance, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def predict_classification(training_data, test_instance, k):
    """Predict the class of a test instance based on majority voting."""
    neighbors = get_neighbors(training_data, test_instance, k)
    output_classes = [neighbor[-1] for neighbor in neighbors]
    class_count = {}
    for cls in output_classes:
        if cls in class_count:
            class_count[cls] += 1
        else:
            class_count[cls] = 1
    prediction = max(class_count, key=class_count.get)
    return prediction


training_data = [
    [2.7, 2.5, "Yellow"],
    [1.3, 1.8, "Yellow"],
    [3.6, 3.2, "Blue"],
    [4.4, 3.8, "Blue"],
    [3.9, 4.1, "Blue"]
]

test_instance = [4.0, 1.0]
k = 5

predicted_label = predict_classification(training_data, test_instance, k)
print(f"Predicted Class: {predicted_label}")