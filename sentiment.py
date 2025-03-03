import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import warnings
import joblib
import time

warnings.filterwarnings('ignore')

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK resource download failed but continuing...")

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
    ["Slept like a baby and woke up energized. Great day ahead!", "Low"],
    ["Feeling a bit anxious about tomorrow's presentation, but I'm prepared.", "Medium"],
    ["Insomnia is killing me. Work pressure is unbearable right now.", "High"],
    ["Kids are on vacation, so the house is peaceful. Getting good sleep.", "Low"],
    ["Some work challenges today, but nothing I can't handle. Sleep was decent.", "Medium"],
    ["Constant arguments at home. Haven't slept properly in days. Feeling horrible.", "High"],
    ["Went for a run, meditated, and had a nutritious meal. Feeling excellent!", "Low"],
    ["Slightly stressed about finances, but managing. Sleep is okay most nights.", "Medium"],
    ["Having panic attacks again. Sleep is disrupted. Can't focus on anything.", "High"],
    ["Vacation mode! Relaxed and well-rested. No worries at all.", "Low"]
]

df = pd.DataFrame(data, columns=['JournalEntry', 'StressLevel'])

print("Class distribution:")
print(df['StressLevel'].value_counts())
print("\n")

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_features(text):
    features = {}
    sleep_words = ['sleep', 'slept', 'rest', 'tired', 'exhausted', 'insomnia', 'nap']
    features['sleep_mentions'] = sum(1 for word in sleep_words if word in text.lower())
    stress_words = ['stress', 'anxious', 'anxiety', 'worry', 'overwhelm', 'pressure']
    features['stress_mentions'] = sum(1 for word in stress_words if word in text.lower())
    positive_words = ['great', 'good', 'happy', 'joy', 'peaceful', 'calm', 'relax', 'positive']
    features['positive_mentions'] = sum(1 for word in positive_words if word in text.lower())
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['exclamation_count'] = text.count('!')
    return features

df['CleanedText'] = df['JournalEntry'].apply(preprocess_text)
feature_df = df['JournalEntry'].apply(lambda x: pd.Series(extract_features(x)))
df = pd.concat([df, feature_df], axis=1)

def visualize_features():
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.countplot(x='StressLevel', data=df)
    plt.title('Distribution of Stress Levels')
    plt.subplot(2, 2, 2)
    feature_means = df.groupby('StressLevel')[['sleep_mentions', 'stress_mentions', 'positive_mentions']].mean()
    feature_means.plot(kind='bar', ax=plt.gca())
    plt.title('Feature Averages by Stress Level')
    plt.subplot(2, 2, 3)
    sns.boxplot(x='StressLevel', y='word_count', data=df)
    plt.title('Word Count by Stress Level')
    plt.subplot(2, 2, 4)
    numeric_cols = ['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.savefig('feature_analysis.png')
    plt.close()
    print("Feature visualization saved to 'feature_analysis.png'")

X_text = df['CleanedText']
X_features = df[['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count']]
y = df['StressLevel']

X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_text_train)}")
print(f"Test set size: {len(X_text_test)}")

text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 3), min_df=2, use_idf=True, sublinear_tf=True))
])

feature_selection = Pipeline([
    ('selector', SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=1000)))
])

pipelines = {
    'LogisticRegression': Pipeline([
        ('features', ColumnTransformer([
            ('text', text_pipeline, 'CleanedText'),
            ('manual', 'passthrough', ['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count'])
        ])),
        ('selection', feature_selection),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ]),
    'RandomForest': Pipeline([
        ('features', ColumnTransformer([
            ('text', text_pipeline, 'CleanedText'),
            ('manual', 'passthrough', ['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count'])
        ])),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('features', ColumnTransformer([
            ('text', text_pipeline, 'CleanedText'),
            ('manual', 'passthrough', ['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count'])
        ])),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('features', ColumnTransformer([
            ('text', text_pipeline, 'CleanedText'),
            ('manual', 'passthrough', ['sleep_mentions', 'stress_mentions', 'positive_mentions', 'text_length', 'word_count', 'exclamation_count'])
        ])),
        ('classifier', SVC(probability=True, class_weight='balanced', random_state=42))
    ])
}

train_data = pd.DataFrame({
    'CleanedText': X_text_train,
    'sleep_mentions': X_features_train['sleep_mentions'],
    'stress_mentions': X_features_train['stress_mentions'],
    'positive_mentions': X_features_train['positive_mentions'],
    'text_length': X_features_train['text_length'],
    'word_count': X_features_train['word_count'],
    'exclamation_count': X_features_train['exclamation_count']
})

test_data = pd.DataFrame({
    'CleanedText': X_text_test,
    'sleep_mentions': X_features_test['sleep_mentions'],
    'stress_mentions': X_features_test['stress_mentions'],
    'positive_mentions': X_features_test['positive_mentions'],
    'text_length': X_features_test['text_length'],
    'word_count': X_features_test['word_count'],
    'exclamation_count': X_features_test['exclamation_count']
})

param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear', 'saga'],
        'features__text__tfidf__max_features': [500, 1000]
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'features__text__tfidf__max_features': [500, 1000]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'features__text__tfidf__max_features': [500, 1000]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'features__text__tfidf__max_features': [500, 1000]
    }
}

results = {}
best_models = {}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid_search.fit(train_data, y_train)
    best_models[name] = grid_search.best_estimator_
    y_pred = best_models[name].predict(test_data)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        'best_params': grid_search.best_params_,
        'training_time': time.time() - start_time
    }
    print(f"{name} Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {results[name]['accuracy']:.4f}")
    print(f"Training time: {results[name]['training_time']:.2f} seconds")
    print(classification_report(y_test, y_pred, zero_division=0))

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = best_models[best_model_name]

print(f"\nBest performing model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

def plot_confusion_matrix(model_name, model, test_data, y_test):
    y_pred = model.predict(test_data)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    print(f"Confusion matrix for {model_name} saved to 'confusion_matrix_{model_name}.png'")

joblib.dump(best_model, 'stress_predictor_model.joblib')
print("Best model saved as 'stress_predictor_model.joblib'")

def predict_stress_level(text, model=best_model):
    features = extract_features(text)
    cleaned_text = preprocess_text(text)
    input_data = pd.DataFrame({
        'CleanedText': [cleaned_text],
        'sleep_mentions': [features['sleep_mentions']],
        'stress_mentions': [features['stress_mentions']],
        'positive_mentions': [features['positive_mentions']],
        'text_length': [features['text_length']],
        'word_count': [features['word_count']],
        'exclamation_count': [features['exclamation_count']]
    })
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
    prob_dict = {cls: probabilities[idx] * 100 for cls, idx in class_indices.items()}
    explanation = []
    if features['sleep_mentions'] > 0:
        explanation.append(f"- Sleep references detected ({features['sleep_mentions']} mentions)")
    if features['stress_mentions'] > 0:
        explanation.append(f"- Stress indicators detected ({features['stress_mentions']} mentions)")
    if features['positive_mentions'] > 0:
        explanation.append(f"- Positive sentiment detected ({features['positive_mentions']} mentions)")
    word_count_explanation = ""
    if features['word_count'] < 10:
        word_count_explanation = "Short entry (may indicate less detailed expression)"
    elif features['word_count'] > 25:
        word_count_explanation = "Detailed entry (more context for analysis)"
    if word_count_explanation:
        explanation.append(f"- {word_count_explanation}")
    return {
        'text': text,
        'prediction': prediction,
        'confidence': {k: f"{v:.1f}%" for k, v in prob_dict.items()},
        'explanation': explanation,
        'features': features
    }

example_entries = [
    "Slept well and had a productive day. No major issues.",
    "Can't focus at all. Worried about everything and haven't slept in days.",
    "A bit tired today but otherwise doing fine. Managed to get some work done.",
    "Just got back from vacation feeling refreshed and energized!",
    "Mixed feelings today. Some work stress but also had a nice lunch with friends."
]

print("\nExample Predictions:")
for entry in example_entries:
    result = predict_stress_level(entry)
    print(f"\nEntry: {result['text']}")
    print(f"Predicted stress level: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print("Explanation:")
    for exp in result['explanation']:
        print(exp)
    print("-" * 50)

def interactive_predict():
    print("\n=== Stress Level Predictor ===")
    while True:
        user_input = input("\nPlease describe how you're feeling (write a complete sentence), or type 'quit' to exit: ")
        if user_input.lower() == 'quit':
            break
        result = predict_stress_level(user_input)
        print(f"\nPredicted stress level: {result['prediction']}")
        print(f"Confidence:")
        for level, confidence in result['confidence'].items():
            print(f"  {level}: {confidence}")
        print("\nExplanation:")
        if result['explanation']:
            for exp in result['explanation']:
                print(exp)
        else:
            print("No specific indicators detected in your text.")
        print("\nKeep in mind this is just an AI prediction and not a medical assessment.")

if __name__ == "__main__":
    interactive_predict()
