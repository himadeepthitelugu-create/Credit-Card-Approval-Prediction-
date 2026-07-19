import sys
sys.path.append('src')

from generate_data import generate_credit_data
from train_model import train_model
from predict import CreditApprovalPredictor


def get_valid_input(prompt, valid_options=None, input_type=str):
    """Get validated user input."""
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                print("Input cannot be empty. Please try again.")
                continue
            
            if valid_options and value not in valid_options:
                print(f"Invalid input. Options: {', '.join(valid_options)}")
                continue
            
            if input_type == int:
                return int(value)
            elif input_type == float:
                return float(value)
            return value
            
        except ValueError:
            print("Invalid number. Please try again.")


def interactive_predict():
    """Interactive prediction mode."""
    print("\n" + "=" * 60)
    print("   CREDIT CARD APPROVAL PREDICTOR")
    print("=" * 60)
    
    predictor = CreditApprovalPredictor()
    
    print("\nPlease enter applicant details:\n")
    
    applicant = {
        'Gender': get_valid_input("Gender (Male/Female): ", ['Male', 'Female']),
        'Age': get_valid_input("Age (18-70): ", input_type=int),
        'Married': get_valid_input("Married (Yes/No): ", ['Yes', 'No']),
        'Dependents': get_valid_input("Dependents (0/1/2/3+): ", ['0', '1', '2', '3+']),
        'Education': get_valid_input("Education (Graduate/Not Graduate): ", ['Graduate', 'Not Graduate']),
        'Self_Employed': get_valid_input("Self Employed (Yes/No): ", ['Yes', 'No']),
        'ApplicantIncome': get_valid_input("Applicant Monthly Income: ", input_type=int),
        'CoapplicantIncome': get_valid_input("Co-applicant Monthly Income: ", input_type=int),
        'LoanAmount': get_valid_input("Loan Amount: ", input_type=int),
        'Loan_Amount_Term': get_valid_input("Loan Term in months (360/180/240/120/60): ", 
                                           ['360', '180', '240', '120', '60'], int),
        'Credit_History': get_valid_input("Credit History (1=Good, 0=Bad): ", ['0', '1'], int),
        'Property_Area': get_valid_input("Property Area (Urban/Semiurban/Rural): ", 
                                        ['Urban', 'Semiurban', 'Rural'])
    }
    
    result = predictor.predict_single(**applicant)
    
    print("\n" + "=" * 60)
    print("   PREDICTION RESULT")
    print("=" * 60)
    
    if result['approved']:
        print("   STATUS: APPROVED")
    else:
        print("   STATUS: REJECTED")
    
    print(f"   Approval Probability: {result['approval_probability']:.2%}")
    print(f"   Rejection Probability: {result['rejection_probability']:.2%}")
    print("=" * 60)
    
    prob = result['approval_probability']
    if prob >= 0.9:
        print("   Risk Level: LOW")
    elif prob >= 0.7:
        print("   Risk Level: MEDIUM")
    else:
        print("   Risk Level: HIGH")
    print("=" * 60)


def main():
    """Main menu."""
    while True:
        print("\n" + "=" * 60)
        print("   CREDIT CARD APPROVAL PREDICTION SYSTEM")
        print("=" * 60)
        print("   1. Generate Dataset")
        print("   2. Train Model")
        print("   3. Make Prediction")
        print("   4. Evaluate Model")
        print("   5. Exit")
        print("=" * 60)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            n = input("Number of samples (default 5000): ").strip()
            n = int(n) if n else 5000
            generate_credit_data(n_samples=n)
            
        elif choice == '2':
            train_model()
            
        elif choice == '3':
            interactive_predict()
            
        elif choice == '4':
            from evaluate import evaluate_model
            evaluate_model()
            
        elif choice == '5':
            print("\nGoodbye!")
            break
            
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()