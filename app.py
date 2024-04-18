import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

pipeline = pickle.load(open('pickled files\pipeline.pkl', 'rb'))
model = pickle.load(open('pickled files\best_model_object.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])

def predict():
    data = request.json['data']
    print(data)
    
    new_data = np.array(list(data.values()).reshape(1,-1)

    scaled_data = pipeline.transform(new_data)

    output = model.predict(scaled_data)
    print(output)

    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)    