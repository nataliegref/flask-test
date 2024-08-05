from flask import Flask, jsonify
from flask_cors import CORS

import pandas as pd 
# from scipy.stats import t

# from sklearn.metrics import mean_squared_error

app = Flask(__name__)
CORS(app) 


def run_average_model():
    # model = get_model("average_demand_trained_model_v1.pkl")

    columns = ['Weekday','Season_Fall','Season_Spring','Season_Summer','Season_Winter',
       'Size', 'Fraction']
    data = {0:[0,0,0,0,1,1000,0.9]}

    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    predictions = df['Size']

    # predictions = model.predict(df)
    return predictions[0]

@app.route('/')
def home():
    return 'Hello, World!'

@app.route("/model", methods=['GET'])
def return_home():
    average = run_average_model()
    return jsonify({
        'message': "Hello! "+str(average)
    })

@app.route('/about')
def about():
    return 'About'

if __name__ == "__main__":
    app.run(port=8000) 