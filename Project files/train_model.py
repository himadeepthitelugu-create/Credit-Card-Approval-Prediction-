import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(data_path='data/credit_data.csv', model_dir='models'):
    """Train and save the credit approval prediction model."""
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    df_processed = df.copy()
    label_encoders = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    X = df_processed.drop('Loan_Status', axis=1)
    y = df_processed['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    numerical_cols = ['Age', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance.to_string(index=False))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, f'{model_dir}/rf_model.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(label_encoders, f'{model_dir}/label_encoders.pkl')
    
    print(f"\nModels saved to '{model_dir}/'")
    
    return rf_model, scaler, label_encoders


if __name__ == "__main__":
    train_model()