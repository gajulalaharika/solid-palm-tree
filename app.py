from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

def predict(values):

    final_model = joblib.load('models/final.pkl')
    values = np.asarray(values).reshape(1, -1)
    print("values:",values)
    return final_model.predict(values)[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            print("dict:",to_predict_dict)
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print("list:", to_predict_list)
            pred = predict(to_predict_list)
    except Exception as e:
        print(e)
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)
