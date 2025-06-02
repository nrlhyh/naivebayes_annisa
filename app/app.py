from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan encoder
model = joblib.load('model/model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[col]) for col in [
            'kecerahan', 'kekeruhan', 'ph', 'suhu', 'salinitas',
            'TSS', 'BOD5', 'Do', 'M&L', 'Coliform', 'NO3N', 'Orthophospate'
        ]]
        prediction = model.predict([features])[0]
        label = label_encoder.inverse_transform([prediction])[0]
        return render_template('index.html', result=label)
    except Exception as e:
        return render_template('index.html', result="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
