from flask import Flask, jsonify

app = Flask(__name__)

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