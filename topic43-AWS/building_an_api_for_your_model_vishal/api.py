import pickle
import flask
import json
import numpy as np

#initalize app
app = flask.Flask(__name__)
#initialize model outside of route so it doesn't have to load everytime it recieves a request
model = pickle.load( open('model.pkl','rb'))

#when the route "/" recieves a request the function hello is run
@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict", methods=["POST"])
def pred():
    if flask.request.method == "POST":
        years = np.float(flask.request.form['years'])
        prediction = model.predict([[years]])
        data = {}
        data['predictions'] = prediction[0]
        return flask.jsonify(data)
#run app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
