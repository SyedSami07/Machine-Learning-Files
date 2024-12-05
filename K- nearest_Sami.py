import numpy as np 
from collections import Counter

def knn_predict(training_data, training_labels, test_data, neighbors=3):
    predictions = []
    for point in test_data:
        distances = [np.linalg.norm(point - data_point) for data_point in training_data]
        nearest_labels = [training_labels[i] for i in np.argsort(distances)[:neighbors]]
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    return predictions

if __name__ == "__main__":
    training_data = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
    training_labels = np.array([0, 0, 0, 1, 1, 1])
    test_data = np.array([[5, 5]])
    predictions = knn_predict(training_data, training_labels, test_data, neighbors=3)
    print("Predicted class:", predictions[0])
