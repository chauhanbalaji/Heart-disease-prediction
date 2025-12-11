import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\BALAJI\OneDrive\Documents\TNS\heart_disease_dataset.csv")

# Features (X) and target (y)
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# --- Preprocessing Pipeline ---
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Combine preprocessing + model into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model with preprocessing
pipeline.fit(X_train, y_train)

# Save trained pipeline
joblib.dump(pipeline, 'logistic_regression_pipeline.pkl')

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
