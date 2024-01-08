import flask
from flask import render_template
import sklearn
import numpy as np
import pandas as pd
import tensorflow
import random

app = flask.Flask(__name__, template_folder = 'templates')


@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')


    if flask.request.method == 'POST':
        loaded_model = tensorflow.keras.models.load_model('my_model.h5')
        total = []

        if flask.request.form['submit_button'] == 'Прогноз':
            parametrs = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12']
            for number in parametrs:
                total.append(float(flask.request.form[number]))
        else:
            #X_test = pd.read_csv('X_test.csv').to_numpy(dtype = 'float')
            #total = random.choice(X_test)
            total = [random.uniform(0, 2000) for _ in range(12)]
        total = np.array(total).reshape(1, 12)    
        y_pred = loaded_model.predict(total)[0][0]
        return render_template('main.html', result = y_pred)

if __name__ == '__main__':
    app.run()