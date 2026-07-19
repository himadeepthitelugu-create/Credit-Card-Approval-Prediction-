from flask import Flask, render_template_string, request
import sys
sys.path.append('src')
from predict import CreditApprovalPredictor

app = Flask(__name__)

HTML_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Approval Predictor</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }
        body { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            margin: 0; 
            padding: 20px;
        }
        .container { 
            background: white; 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
            width: 100%; 
            max-width: 500px;
        }
        h1 { 
            text-align: center; 
            color: #333; 
            margin-bottom: 30px; 
            font-size: 28px;
        }
        .form-group { 
            margin-bottom: 18px; 
        }
        label { 
            display: block; 
            margin-bottom: 6px; 
            color: #555; 
            font-weight: 600; 
            font-size: 14px;
        }
        input, select { 
            width: 100%; 
            padding: 12px 15px; 
            border: 2px solid #e0e0e0; 
            border-radius: 10px; 
            font-size: 15px; 
            transition: border-color 0.3s;
        }
        input:focus, select:focus { 
            outline: none; 
            border-color: #667eea; 
        }
        button { 
            width: 100%; 
            padding: 15px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border: none; 
            border-radius: 10px; 
            font-size: 18px; 
            font-weight: bold; 
            cursor: pointer; 
            margin-top: 10px;
            transition: transform 0.2s;
        }
        button:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        .result { 
            margin-top: 25px; 
            padding: 25px; 
            border-radius: 15px; 
            text-align: center; 
            display: none;
        }
        .result.approved { 
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
            color: white; 
        }
        .result.rejected { 
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
            color: white; 
        }
        .result h2 { 
            margin: 0 0 10px 0; 
            font-size: 32px;
        }
        .result p { 
            margin: 5px 0; 
            font-size: 16px;
        }
        .probability { 
            font-size: 36px !important; 
            font-weight: bold; 
            margin: 15px 0 !important;
        }
        .risk { 
            display: inline-block; 
            padding: 8px 20px; 
            border-radius: 20px; 
            background: rgba(255,255,255,0.3); 
            margin-top: 10px; 
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💳 Credit Card Approval</h1>
        <form method="POST">
            <div class="form-group">
                <label>Gender</label>
                <select name="Gender" required>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>Age</label>
                <input type="number" name="Age" min="18" max="70" value="35" required>
            </div>
            <div class="form-group">
                <label>Married</label>
                <select name="Married" required>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Dependents</label>
                <select name="Dependents" required>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3+">3+</option>
                </select>
            </div>
            <div class="form-group">
                <label>Education</label>
                <select name="Education" required>
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                </select>
            </div>
            <div class="form-group">
                <label>Self Employed</label>
                <select name="Self_Employed" required>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Applicant Monthly Income</label>
                <input type="number" name="ApplicantIncome" value="8000" required>
            </div>
            <div class="form-group">
                <label>Co-applicant Monthly Income</label>
                <input type="number" name="CoapplicantIncome" value="3000" required>
            </div>
            <div class="form-group">
                <label>Loan Amount</label>
                <input type="number" name="LoanAmount" value="150000" required>
            </div>
            <div class="form-group">
                <label>Loan Term (months)</label>
                <select name="Loan_Amount_Term" required>
                    <option value="360" selected>360</option>
                    <option value="240">240</option>
                    <option value="180">180</option>
                    <option value="120">120</option>
                    <option value="60">60</option>
                </select>
            </div>
            <div class="form-group">
                <label>Credit History</label>
                <select name="Credit_History" required>
                    <option value="1">Good (1)</option>
                    <option value="0">Bad (0)</option>
                </select>
            </div>
            <div class="form-group">
                <label>Property Area</label>
                <select name="Property_Area" required>
                    <option value="Urban">Urban</option>
                    <option value="Semiurban">Semiurban</option>
                    <option value="Rural">Rural</option>
                </select>
            </div>
            <button type="submit">🔮 Predict Approval</button>
        </form>

        {% if result %}
        <div class="result {{ 'approved' if result.approved else 'rejected' }}" style="display: block;">
            <h2>{{ '✅ APPROVED' if result.approved else '❌ REJECTED' }}</h2>
            <p class="probability">{{ "%.1f"|format(result.approval_probability * 100) }}%</p>
            <p>Approval Probability</p>
            <p>Rejection Probability: {{ "%.1f"|format(result.rejection_probability * 100) }}%</p>
            <span class="risk">
                {% if result.approval_probability >= 0.9 %}
                    🟢 LOW RISK
                {% elif result.approval_probability >= 0.7 %}
                    🟡 MEDIUM RISK
                {% else %}
                    🔴 HIGH RISK
                {% endif %}
            </span>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        predictor = CreditApprovalPredictor()
        data = {
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Married': request.form['Married'],
            'Dependents': request.form['Dependents'],
            'Education': request.form['Education'],
            'Self_Employed': request.form['Self_Employed'],
            'ApplicantIncome': int(request.form['ApplicantIncome']),
            'CoapplicantIncome': int(request.form['CoapplicantIncome']),
            'LoanAmount': int(request.form['LoanAmount']),
            'Loan_Amount_Term': int(request.form['Loan_Amount_Term']),
            'Credit_History': int(request.form['Credit_History']),
            'Property_Area': request.form['Property_Area']
        }
        result = predictor.predict_single(**data)
    return render_template_string(HTML_PAGE, result=result)

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 Starting Credit Card Approval Web App")
    print("=" * 60)
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, port=8080)