from flask import Flask,jsonify,render_template,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

mod = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    features = [int(x) for x in request.form.values()]
    final=[np.array(features)]
    prediction = mod.predict(final)

    output = round(prediction[0], 2)
    if output>0.5:
        return render_template('index2.html',prediction_text = 'Yes, this user will pay next month')
    else:
        return render_template('index2.html',prediction_text = 'No, this user will not pay next month')


if __name__ == "__main__":
    app.run(debug=True)              


