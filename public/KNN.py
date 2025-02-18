import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def process_data(data):
    """Process the dataset to separate features and target variable."""
    # Assume the last column is the target variable
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Check for NaN values in features and target
    if X.isnull().any().any():
        print("Warning: Features contain NaN values. They will be dropped.")
        X = X.dropna()  # Drop rows with NaN in features
        y = y[X.index]  # Align y with the dropped rows

    if y.isnull().any():
        print("Warning: Target contains NaN values. They will be dropped.")
        y = y.dropna()  # Drop NaN values from target

    # Convert categorical variables to numeric using Label Encoding
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Convert target variable if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y

def check_class_distribution(y):
    """Check the distribution of classes in the target variable."""
    class_counts = pd.Series(y).value_counts()
    print("Class distribution in the dataset:")
    print(class_counts)
    return class_counts

def plot_confusion_matrix(y_test, y_pred):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')
    plt.show()

def plot_k_vs_accuracy(X, y):
    """Plot accuracy vs different values of K."""
    neighbors = range(1, 11)
    accuracies = []

    for k in neighbors:
        model = KNeighborsClassifier(n_neighbors=k)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.plot(neighbors, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.title('K vs Accuracy')
    plt.xticks(neighbors)
    plt.grid(True)
    plt.show()

def main(file_path):
    """Main function to load data, split it, train KNN, and evaluate."""

    # Load the dataset
    data = load_data(file_path)

    # Process the data
    X, y = process_data(data)

    # Check class distribution
    class_counts = check_class_distribution(y)

    # Check if any class has less than 2 members
    if class_counts.min() < 2:
        print("Error: At least one class has fewer than 2 members. Please check your dataset.")
        return

    # Split the data into training and testing sets while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create KNN model with k=2 neighbors
    k = 2
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Plot K vs Accuracy
    plot_k_vs_accuracy(X, y)

# Run the main function with the specified CSV file
main('knn.csv')
