
import numpy as np
from flask import Flask, render_template, request, jsonify
#from flask_restful import Api, Resource
import pickle
import weakref
app = Flask(__name__)
#api = Api(app)
m = pickle.load(open("model.pkl", "rb"))
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = m.predict(features)
    p=float(prediction)
    return render_template("home.html", prediction_text = "{}".format(p))




if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
