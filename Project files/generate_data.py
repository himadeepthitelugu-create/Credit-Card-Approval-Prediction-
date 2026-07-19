import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_credit_data(n_samples=5000, output_path='data/credit_data.csv'):
    """Generate synthetic credit approval data."""
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.5, 0.25, 0.15, 0.1]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'ApplicantIncome': np.random.lognormal(8.2, 0.6, n_samples).astype(int),
        'CoapplicantIncome': np.random.lognormal(7.5, 0.8, n_samples).astype(int),
        'LoanAmount': np.random.lognormal(11.5, 0.4, n_samples).astype(int),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 120, 60], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        'Credit_History': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.35, 0.25]),
    }
    
    df = pd.DataFrame(data)
    
    def generate_approval(row):
        score = 0
        score += (row['Age'] - 18) / 52 * 15
        score += np.log(row['ApplicantIncome'] + 1) * 8
        score += row['Credit_History'] * 25
        score += (row['Education'] == 'Graduate') * 10
        score += (row['Married'] == 'Yes') * 5
        score += np.log(row['LoanAmount'] + 1) * (-3)
        score += (row['Property_Area'] == 'Urban') * 5
        score += (row['Self_Employed'] == 'No') * 3
        score += np.random.normal(0, 8)
        return 1 if score > 50 else 0
    
    df['Loan_Status'] = df.apply(generate_approval, axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated: {output_path}")
    print(f"Samples: {len(df)}")
    print(f"Approval Rate: {df['Loan_Status'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_credit_data()