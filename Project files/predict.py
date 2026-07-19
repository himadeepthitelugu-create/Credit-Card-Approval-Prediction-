import pandas as pd
import numpy as np
import joblib

class CreditApprovalPredictor:
    """Credit Card Approval Prediction class."""
    
    def __init__(self, model_dir='models'):
        self.model = joblib.load(f'{model_dir}/rf_model.pkl')
        self.scaler = joblib.load(f'{model_dir}/scaler.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        self.numerical_cols = ['Age', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        self.categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    def preprocess(self, df):
        """Preprocess input data."""
        df_processed = df.copy()
        
        for col in self.categorical_cols:
            if col in df_processed.columns:
                le = self.label_encoders[col]
                df_processed[col] = df_processed[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        df_scaled = df_processed.copy()
        df_scaled[self.numerical_cols] = self.scaler.transform(df_processed[self.numerical_cols])
        
        return df_scaled
    
    def predict(self, df):
        """Predict approval status."""
        df_processed = self.preprocess(df)
        predictions = self.model.predict(df_processed)
        probabilities = self.model.predict_proba(df_processed)
        return predictions, probabilities
    
    def predict_single(self, **kwargs):
        """Predict for a single applicant."""
        df = pd.DataFrame([kwargs])
        pred, prob = self.predict(df)
        return {
            'approved': bool(pred[0]),
            'approval_probability': float(prob[0][1]),
            'rejection_probability': float(prob[0][0])
        }


def main():
    """Example usage."""
    predictor = CreditApprovalPredictor()
    
    applicant = {
        'Gender': 'Male',
        'Age': 35,
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 8000,
        'CoapplicantIncome': 3000,
        'LoanAmount': 150000,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 'Urban'
    }
    
    result = predictor.predict_single(**applicant)
    
    print("=" * 50)
    print("CREDIT CARD APPROVAL PREDICTION")
    print("=" * 50)
    for key, value in applicant.items():
        print(f"   {key}: {value}")
    print("-" * 50)
    print(f"   Approved: {result['approved']}")
    print(f"   Approval Probability: {result['approval_probability']:.2%}")
    print(f"   Rejection Probability: {result['rejection_probability']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
