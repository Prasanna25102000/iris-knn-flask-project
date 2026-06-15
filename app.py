from flask import Flask, render_template, request
from model import predict_flower

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = predict_flower(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    return render_template(
        'index.html',
        prediction=result
    )

if __name__ == '__main__':
    app.run(debug=True)