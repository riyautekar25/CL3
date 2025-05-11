import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Step 1: Generate Dummy Dataset
def generate_dummy_data(samples=100, features=10):
    data = np.random.rand(samples, features)
    labels = np.random.randint(0, 2, size=samples)
    return data, labels


# Step 2: Define a Detector Class
class Detector:
    def __init__(self, vector, label):
        self.vector = vector  # the antibody
        self.label = label    # its class label


# Step 3: Clonal Selection Algorithm (CSA)
class ClonalSelectionClassifier:
    def __init__(self, num_detectors=10, hypermutation_rate=0.1):
        self.num_detectors = num_detectors
        self.hypermutation_rate = hypermutation_rate
        self.detectors = []

    def train(self, X, y):
        indices = np.random.choice(len(X), self.num_detectors, replace=False)
        for i in indices:
            detector = Detector(X[i], y[i])
            self.detectors.append(detector)

    def predict(self, X):
        predictions = []
        for sample in X:
            # Find the detector with minimum distance
            distances = [np.linalg.norm(sample - d.vector) for d in self.detectors]
            best_index = np.argmin(distances)
            best_label = self.detectors[best_index].label
            predictions.append(best_label)
        return np.array(predictions)

    def mutate(self):
        # Hypermutation: apply small noise to detectors
        for d in self.detectors:
            mutation = np.random.normal(0, self.hypermutation_rate, size=d.vector.shape)
            d.vector += mutation
            d.vector = np.clip(d.vector, 0, 1)  # keep values in [0,1]


# Step 4: Main Code
# Generate dataset
data, labels = generate_dummy_data(samples=200, features=10)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize classifier
clf = ClonalSelectionClassifier(num_detectors=20, hypermutation_rate=0.05)
clf.train(X_train, y_train)
clf.mutate()  # Optional mutation step

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"CSA Classifier Accuracy: {accuracy * 100:.2f}%")
