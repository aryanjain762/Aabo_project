import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the data
data = pd.DataFrame({
    'customer_id': ['CUST1', 'CUST1', 'CUST1', 'CUST1', 'CUST1', 'CUST1', 
                   'CUST2', 'CUST2', 'CUST2', 'CUST2', 'CUST2', 'CUST2',
                   'CUST3', 'CUST3', 'CUST3'],
    'date': ['24/02/25', '23/02/25', '22/02/25', '21/02/25', '20/02/25', '19/02/25',
            '24/02/25', '23/02/25', '22/02/25', '21/02/25', '20/02/25', '19/02/25',
            '12/02/25', '11/02/25', '10/02/25'],
    'sleep_duration': [570, 421, 387, 593, 321, 358,
                      350, 434, 541, 391, 563, 349,
                      353, 517, 569],
    'steps': [10191, 5466, 13322, 7433, 11396, 7558,
             6899, 10393, 12513, 10486, 13226, 8943,
             8843, 10675, 12629],
    'resting_heart_rate': [78, 72, 71, 77, 61, 59,
                          72, 53, 70, 77, 63, 53,
                          71, 51, 72]
})

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%y')

# Sort by customer_id and date
data = data.sort_values(['customer_id', 'date'])

# Extract day of week (0=Monday, 6=Sunday)
data['day_of_week'] = data['date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Enhanced Feature Engineering
# Previous day metrics
data['prev_sleep'] = data.groupby('customer_id')['sleep_duration'].shift(1)
data['prev_steps'] = data.groupby('customer_id')['steps'].shift(1)
data['prev_heart_rate'] = data.groupby('customer_id')['resting_heart_rate'].shift(1)

# Rolling statistics (mean, min, max, std) with different windows
for window in [2, 3]:
    # Rolling averages
    data[f'sleep_roll_mean_{window}d'] = data.groupby('customer_id')['sleep_duration'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    data[f'steps_roll_mean_{window}d'] = data.groupby('customer_id')['steps'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    data[f'heart_roll_mean_{window}d'] = data.groupby('customer_id')['resting_heart_rate'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    
    # Rolling standard deviation (captures variability)
    data[f'sleep_roll_std_{window}d'] = data.groupby('customer_id')['sleep_duration'].transform(
        lambda x: x.rolling(window=window, min_periods=2).std().shift(1).fillna(0)
    )
    data[f'steps_roll_std_{window}d'] = data.groupby('customer_id')['steps'].transform(
        lambda x: x.rolling(window=window, min_periods=2).std().shift(1).fillna(0)
    )
    
    # Min and max values (captures range)
    data[f'sleep_roll_min_{window}d'] = data.groupby('customer_id')['sleep_duration'].transform(
        lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
    )
    data[f'sleep_roll_max_{window}d'] = data.groupby('customer_id')['sleep_duration'].transform(
        lambda x: x.rolling(window=window, min_periods=1).max().shift(1)
    )

# Rate of change features (captures trends)
data['sleep_change'] = data.groupby('customer_id')['sleep_duration'].diff().fillna(0)
data['steps_change'] = data.groupby('customer_id')['steps'].diff().fillna(0)
data['heart_change'] = data.groupby('customer_id')['resting_heart_rate'].diff().fillna(0)

# Interaction features
data['steps_per_hr'] = data['steps'] / data['resting_heart_rate']
data['prev_steps_per_hr'] = data['prev_steps'] / data['prev_heart_rate'].replace(0, 1)

# Drop rows with NaN values
data_processed = data.dropna(subset=['prev_sleep']).copy()

# Select features
features = [
    'prev_sleep', 'prev_steps', 'prev_heart_rate', 
    'sleep_roll_mean_2d', 'steps_roll_mean_2d', 'heart_roll_mean_2d',
    'sleep_roll_mean_3d', 'steps_roll_mean_3d', 'heart_roll_mean_3d',
    'sleep_roll_std_2d', 'steps_roll_std_2d',
    'sleep_roll_min_3d', 'sleep_roll_max_3d',
    'sleep_change', 'steps_change', 'heart_change',
    'prev_steps_per_hr', 'is_weekend', 'day_of_week'
]

X = data_processed[features]
y = data_processed['sleep_duration']

# Build a pipeline with polynomial features, scaling, and Ridge regression
# (Ridge helps control overfitting with polynomial features)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])

# Use cross-validation to evaluate the model
cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())

# Fine-tune the model with GridSearchCV
param_grid = {
    'poly__degree': [1, 2],
    'ridge__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, 
    scoring='neg_mean_squared_error', verbose=0
)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate on the training data
y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Function to get feature importance after polynomial transformation
def get_feature_importance(model, feature_names):
    # Get the polynomial feature names
    poly = model.named_steps['poly']
    poly_features = PolynomialFeatures(degree=poly.degree, include_bias=False)
    poly_features.fit_transform(np.zeros((1, len(feature_names))))
    poly_feature_names = poly_features.get_feature_names_out(feature_names)
    
    # Get the coefficients from the Ridge model
    ridge = model.named_steps['ridge']
    coefficients = ridge.coef_
    
    # Create a dataframe of features and their coefficients
    feature_importance = pd.DataFrame({
        'Feature': poly_feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    return feature_importance

# Get feature importance
feature_importance = get_feature_importance(best_model, features)

# Function to predict next day's sleep for a customer
def predict_next_day_sleep(customer_id, latest_data, model, feature_names):
    customer_data = latest_data[latest_data['customer_id'] == customer_id].sort_values('date')
    
    if len(customer_data) < 1:
        return None
    
    # Extract the features for prediction
    latest_features = customer_data[feature_names].iloc[-1:].values
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    
    return prediction

# Example: Predict next day's sleep for CUST1
cust1_prediction = predict_next_day_sleep('CUST1', data_processed, best_model, features)

# Print results
print("\nBest Model Parameters:")
print(best_params)

print(f"\nModel Performance:")
print(f"Cross-Validation RMSE: {cv_rmse:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print(f"\nPredicted sleep duration for CUST1's next day: {cust1_prediction:.1f} minutes")

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize the model performance
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Sleep Duration (minutes)')
plt.ylabel('Predicted Sleep Duration (minutes)')
plt.title('Sleep Duration Prediction Model Performance')
plt.tight_layout()

# Visualize top 10 feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance')wa
plt.tight_layout()