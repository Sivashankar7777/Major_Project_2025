#ANN
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('review.csv')

# Assuming the 'Review Text' column exists in the CSV
# If your column has a different name, replace 'Review Text' with the actual column name
# Simplified assumption for this example: create artificial sentiment labels
# Assuming every review with "good", "amazing", "excellent" is positive (label 1), others negative (label 0)
df['Sentiment'] = df['Review Text'].apply(lambda x: 1 if any(word in x.lower() for word in ['good', 'amazing', 'excellent']) else 0)

# Vectorize the 'Review Text' column using TF-IDF
tfidf = TfidfVectorizer(max_features=500)  # Limiting to 500 features for simplicity
X = tfidf.fit_transform(df['Review Text']).toarray()

# Sentiment labels as target
y = df['Sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ANN model
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
plt.matshow(conf_matrix, cmap='Blues', alpha=0.8)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot the loss curve (if available)
try:
    plt.figure()
    plt.plot(model.loss_curve_)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
except:
    print("Loss curve not available.")

# Display accuracy and classification report
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_rep)
