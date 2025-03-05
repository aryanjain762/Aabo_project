import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

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
    ["Too many things to handle today. My sleep has been terrible, and I'm feeling on edge.", "High"],
]

df = pd.DataFrame(data, columns=['JournalEntry', 'StressLevel'])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\d+', '', text)  
    return text

df['CleanedText'] = df['JournalEntry'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['CleanedText'], df['StressLevel'], test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'tfidf__max_features': [500, 1000],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(best_model, 'stress_predictor_model.joblib')
print("Model saved as 'stress_predictor_model.joblib'")

def predict_stress_level(text, model=best_model):
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])[0]
    return prediction

example_entries = [
    "Slept well and had a productive day. No major issues.",
    "Can't focus at all. Worried about everything and haven't slept in days.",
    "A bit tired today but otherwise doing fine. Managed to get some work done."
]

print("\nExample Predictions:")
for entry in example_entries:
    print(f"Entry: {entry}")
    print(f"Predicted Stress Level: {predict_stress_level(entry)}")
    print("-" * 50)
