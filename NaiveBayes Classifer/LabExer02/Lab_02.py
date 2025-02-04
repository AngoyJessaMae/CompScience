def train_naive_bayes(data):
    """ Trains a Naive Bayes classifier manually """
    class_probabilities = {}
    feature_probabilities = {}
    
    # Count occurrences of each class
    total_count = len(data)
    class_counts = {}
    for row in data:
        label = row[-1]  # Last column is the class label
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Compute class probabilities
    for label, count in class_counts.items():
        class_probabilities[label] = count / total_count
    
    # Compute conditional probabilities for each feature given a class
    num_features = len(data[0]) - 1
    
    for feature_idx in range(num_features):
        feature_probabilities[feature_idx] = {}
        for row in data:
            label = row[-1]
            feature_value = row[feature_idx]
            
            if label not in feature_probabilities[feature_idx]:
                feature_probabilities[feature_idx][label] = {}
            
            if feature_value not in feature_probabilities[feature_idx][label]:
                feature_probabilities[feature_idx][label][feature_value] = 0
            
            feature_probabilities[feature_idx][label][feature_value] += 1
    
    # Convert counts to probabilities
    for feature_idx in feature_probabilities:
        for label in feature_probabilities[feature_idx]:
            total = class_counts[label]
            for feature_value in feature_probabilities[feature_idx][label]:
                feature_probabilities[feature_idx][label][feature_value] /= total
    
    return class_probabilities, feature_probabilities

def predict_naive_bayes(test_data, class_probabilities, feature_probabilities):
    """ Predicts the class using the trained Naive Bayes model """
    predictions = []
    
    for test_instance in test_data:
        class_scores = {}
        
        for label in class_probabilities:
            probability = class_probabilities[label]
            
            for feature_idx, feature_value in enumerate(test_instance):
                if feature_value in feature_probabilities[feature_idx][label]:
                    probability *= feature_probabilities[feature_idx][label][feature_value]
                else:
                    probability *= 0  # Laplace smoothing can be added here
            
            class_scores[label] = probability
        
        predictions.append(max(class_scores, key=class_scores.get))
    
    return predictions

def main():
    # Sample dataset
    # Columns: Color, Toughness, Fungus, Appearance, Poisonous
    training_data = [
        ["Red", "Hard", "Yes", "Smooth", "Poisonous"],
        ["Green", "Soft", "No", "Rough", "Not Poisonous"],
        ["Red", "Soft", "Yes", "Smooth", "Poisonous"],
        ["Yellow", "Hard", "No", "Rough", "Not Poisonous"],
        ["Red", "Hard", "Yes", "Rough", "Poisonous"],
        ["Green", "Soft", "Yes", "Smooth", "Not Poisonous"],
        ["Yellow", "Soft", "No", "Smooth", "Not Poisonous"],
        ["Red", "Hard", "No", "Rough", "Poisonous"],
        ["Green", "Hard", "Yes", "Smooth", "Not Poisonous"],
        ["Yellow", "Soft", "Yes", "Rough", "Not Poisonous"],
        ["Red", "Soft", "No", "Smooth", "Poisonous"],
        ["Green", "Hard", "No", "Rough", "Not Poisonous"],
        ["Yellow", "Hard", "Yes", "Smooth", "Not Poisonous"],
        ["Red", "Soft", "Yes", "Rough", "Poisonous"]
    ]

    # Train Naive Bayes Model
    class_probs, feature_probs = train_naive_bayes(training_data)

    # Interactive Test Data Input
    test_data = []
    num_tests = int(input("Enter the number of test instances: "))
    for _ in range(num_tests):
        color = input("Enter Color (Red/Green/Yellow): ")
        toughness = input("Enter Toughness (Hard/Soft): ")
        fungus = input("Enter Fungus (Yes/No): ")
        appearance = input("Enter Appearance (Smooth/Rough): ")
        test_data.append([color, toughness, fungus, appearance])

    # Make Predictions
    predictions = predict_naive_bayes(test_data, class_probs, feature_probs)

    # Print Predictions
    for test_instance, prediction in zip(test_data, predictions):
        print(f"Test Instance {test_instance}: Classified as {prediction}")

if __name__ == "__main__":
    main()