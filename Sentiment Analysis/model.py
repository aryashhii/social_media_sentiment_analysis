import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load the merged dataset
df = pd.read_csv("merged_dataset.csv")

# Remove rows with missing values
df = df.dropna(subset=["clean_text", "sentiment"])

# Display first 5 rows
print(df.head())

# Display column names
print(df.columns)

# Display dataset size
print(df.shape)

# Check for missing values
print("Missing clean_text:", df["clean_text"].isnull().sum())
print("Missing sentiment:", df["sentiment"].isnull().sum())

# Select features and target
X = df["clean_text"]
y = df["sentiment"]

# Display a few samples
print(X.head())
print(y.head())



# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numerical features
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Display the shapes
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)



# Create Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")

# Predict on test data
lr_predictions = lr_model.predict(X_test)

print("Prediction completed!")



print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))

print(classification_report(y_test, lr_predictions))

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())



# Create Naive Bayes model
nb_model = MultinomialNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict
nb_predictions = nb_model.predict(X_test)

# Evaluate
print("\nNaive Bayes Accuracy:",
      accuracy_score(y_test, nb_predictions))

print(classification_report(y_test, nb_predictions))



# Create SVM model
svm_model = LinearSVC()

# Train the model
svm_model.fit(X_train, y_train)

# Predict
svm_predictions = svm_model.predict(X_test)

# Evaluate
print("\nSVM Accuracy:",
      accuracy_score(y_test, svm_predictions))

print(classification_report(y_test, svm_predictions))





# Create comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "Linear SVM"],
    "Accuracy": [
        accuracy_score(y_test, lr_predictions),
        accuracy_score(y_test, nb_predictions),
        accuracy_score(y_test, svm_predictions)
    ],
    "Precision": [
        precision_score(y_test, lr_predictions, average="weighted"),
        precision_score(y_test, nb_predictions, average="weighted"),
        precision_score(y_test, svm_predictions, average="weighted")
    ],
    "Recall": [
        recall_score(y_test, lr_predictions, average="weighted"),
        recall_score(y_test, nb_predictions, average="weighted"),
        recall_score(y_test, svm_predictions, average="weighted")
    ],
    "F1 Score": [
        f1_score(y_test, lr_predictions, average="weighted"),
        f1_score(y_test, nb_predictions, average="weighted"),
        f1_score(y_test, svm_predictions, average="weighted")
    ]
})

print("\nModel Comparison")
print(comparison)



joblib.dump(lr_model, "model.pkl")

print("Model saved successfully!")

joblib.dump(vectorizer, "vectorizer.pkl")

print("Vectorizer saved successfully!")

new_review = ["This product is excellent"]

new_review_vector = vectorizer.transform(new_review)

prediction = lr_model.predict(new_review_vector)

print("Prediction:", prediction[0])