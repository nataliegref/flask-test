from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return 'Hello, World!'

@app.route("/model", methods=['GET'])
def return_home():
    average = 20
    return jsonify({
        'message': "Hello! "+str(average)
    })

@app.route('/about')
def about():
    return 'About'