from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np


# Loading trained model 
model = pickle.load(open('best_model_object.pkl', 'rb'))

# Loading pipeline
scale = pickle.load(open('scale.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get data from Post request
    
    # age = request.form.get('age')
    # workclass = request.form.get('workclass')
    # fnlwgt = request.form.get('fnlwgt')
    # education_num = request.form.get('education_num')
    # marital_status = request.form.get('marital_status')
    # occupation = request.form.get('occupation')
    # relationship = request.form.get('relationship')
    # race = request.form.get('race')
    # sex = request.form.get('sex')
    # hours_per_week = request.form.get('hours_per_week')
    # native_country = request.form.get('native_country')

    data = [x for x in request.form.values()]

    transformed_data = scale.transform(np.array(data).reshape(1, 10))
    
    prediction = model.predict(transformed_data)
    print(prediction[0])

    if prediction==0:
        msg = "Income less than 50  thousand"
    else:
        msg = "Income greater than 50 thousand"

    return render_template('home.html', data=data, msg=msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
