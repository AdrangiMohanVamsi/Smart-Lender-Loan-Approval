from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import pickle

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\vamsi\Downloads\Documentation_Smart\6. Project Executable Files\Flask\loan_prediction.pkl', 'rb'))
scale = pickle.load(open(r'C:\Users\vamsi\Downloads\Documentation_Smart\6. Project Executable Files\Flask\scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    return render_template('input.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    input_feature = [int(x) for x in request.form.values()]
    input_feature = [np.array(input_feature)]
    print(input_feature)
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    data = pd.DataFrame(input_feature, columns=names)
    print(data)

    scaled_data = scale.transform(data)
    print(scaled_data)
    prediction = model.predict(scaled_data)
    print(prediction)
    prediction = int(prediction)
    print(type(prediction))

    if prediction == 0:
        return render_template('output1.html', result='Loan will Not be Approved')
    else:
        return render_template('output.html', result='Loan will be Approved')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False)