from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

filename = 'randfor.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    Area = request.form['Area']
    Perimeter = request.form['Perimeter']
    MajorAxisLength = request.form['MajorAxisLength']
    MinorAxisLength = request.form['MinorAxisLength']

    ConvexArea = request.form['ConvexArea']
    EquivDiameter = request.form['EquivDiameter']
    roundness = request.form['roundness']
    ShapeFactor1 = request.form['ShapeFactor1']
    ShapeFactor2 = request.form['ShapeFactor2']   

    pred = model.predict(np.array([[Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, roundness, ShapeFactor1, ShapeFactor2]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))

#if __name__ == '__main__':
#    app.run

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#if __name__ == '__main__':
#    port = int(os.environ.get('PORT', 5000))
#    app.run(host='0.0.0.0', port=port)