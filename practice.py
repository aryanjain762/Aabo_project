import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Sample data from the provided table
data = [
    ["I couldn't sleep well last night. Feeling exhausted and anxious. Work is overwhelming.", "High"],
    ["Had a great workout today! Feeling pumped and positive. Got decent sleep too.", "Low"],
    ["Didn't get much rest. Kids were up all night. I'm a bit tired but managing okay.", "Medium"],
    ["Feeling really stressed about upcoming deadlines. My sleep was all over the place this week.", "High"],
    ["Had a peaceful day. Managed to catch up on sleep. Feeling relaxed.", "Low"],
    ["I'm okay, just a bit tired. Work was busy but nothing too stressful.", "Medium"],
    ["Another sleepless night. My mind won't stop racing. Everything feels too much right now.", "High"],
    ["Enjoyed a walk in the park. Got a full 8 hours of sleep. Feeling calm and refreshed.", "Low"],
    ["Busy day. Got some sleep but still a bit restless. Managing stress okay.", "Medium"],
    ["Too many things to handle today. My sleep has been terrible, and I'm feeling on edge.", "High"]
]

# Create DataFrame
df = pd.DataFrame(data, columns=['JournalEntry', 'StressLevel'])

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['CleanedText'] = df['JournalEntry'].apply(clean_text)

# Split data into features and target
X = df['CleanedText']
y = df['StressLevel']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42, stratify=y)

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate with cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=3)
print(f"Cross-validation accuracy: {np.mean(cv_scores):.2f}")

# Make predictions on test set
y_pred = pipeline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Example of using the model to predict stress levels from new entries
new_entries = [
    "Slept well and had a productive day. No major issues.",
    "Can't focus at all. Worried about everything and haven't slept in days.",
    "A bit tired today but otherwise doing fine. Managed to get some work done."
]

# Clean the new entries
cleaned_entries = [clean_text(entry) for entry in new_entries]

# Predict stress levels
predictions = pipeline.predict(cleaned_entries)
for entry, prediction in zip(new_entries, predictions):
    print(f"Entry: {entry}")
    print(f"Predicted stress level: {prediction}\n")

# Function to explain the prediction
def explain_prediction(text):
    cleaned = clean_text(text)
    
    # Get the TF-IDF vectorizer and classifier from the pipeline
    tfidf = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    # Transform the text
    features = tfidf.transform([cleaned])
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get coefficients for each class
    class_labels = classifier.classes_
    coefficients = classifier.coef_
    
    # Get top features for each class
    top_features = {}
    for i, label in enumerate(class_labels):
        if len(class_labels) > 2:  # Multiclass case
            class_coef = coefficients[i]
        else:  # Binary case
            class_coef = coefficients[0] if i == 1 else -coefficients[0]
            
        # Get indices of top features
        top_indices = np.argsort(class_coef)[-10:]
        top_features[label] = [(feature_names[j], class_coef[j]) for j in top_indices]
    
    return top_features

# Example of explaining a prediction
explanation = explain_prediction("I'm feeling overwhelmed with work and couldn't sleep")
for label, features in explanation.items():
    print(f"Top features for {label} stress level:")
    for feature, coef in features:
        print(f"  {feature}: {coef:.4f}")
    print()

# Function to predict stress level from user input
def predict_stress_level():
    # Get user input
    user_input = input("Please describe how you're feeling (write a complete sentence): ")
    
    # Clean the input
    cleaned_input = clean_text(user_input)
    
    # Make prediction
    prediction = pipeline.predict([cleaned_input])[0]
    
    # Print results
    print(f"\nBased on your input: '{user_input}'")
    print(f"Predicted stress level: {prediction}")

# Run the prediction function in a loop
while True:
    predict_stress_level()
    
    # Ask if user wants to continue
    continue_input = input("\nWould you like to check another stress level? (yes/no): ")
    if continue_input.lower() != 'yes':
        break