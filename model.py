import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_student_data

def train_and_get_pipeline():
    """
    Loads data, splits it, trains a Random Forest pipeline, 
    and returns the trained pipeline model.
    """
    # 1. Load Data
    df = load_student_data()
    
    # 2. Separate Features (X) and Target (y)
    # We use background info + reading/writing scores to predict math score
    X = df[['gender', 'race/ethnicity', 'parental level of education', 
            'lunch', 'test preparation course', 'reading score', 'writing score']]
    y = df['math score']
    
    # 3. Define Categorical and Numerical features
    categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 
                            'lunch', 'test preparation course']
    
    # 4. Create Preprocessing Pipeline (One-Hot Encoding for text categories)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep reading and writing scores as they are
    )
    
    # 5. Create complete ML Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # 6. Train-Test Split & Fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)
    
    print("Training the AI model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate performance roughly
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Model Trained! R² Scores -> Train: {train_score:.2f}, Test: {test_score:.2f}")
    
    return pipeline, df

if __name__ == "__main__":
    # Test building the model
    model, _ = train_and_get_pipeline()