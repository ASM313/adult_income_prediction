from flask import Flask, request, jsonify, render_template
import pickle


# Loading trained model 
model = pickle.load(open('best_model_object.pkl', 'rb'))

# Loading pipeline
model = pickle.load(open('pipeline.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get data from Post request
    data = request.get_json()
    print(data)
    # Assuming data is a list of lists for a model expecting multiple inputs
    predictions = model.predict(data)
    print(predictions[0])
    # Convert predictions to a list (if they're not already) and send back
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
