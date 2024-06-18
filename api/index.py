from flask import Flask, jsonify
from flask_cors import CORS

import pickle
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd 
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import os

app = Flask(__name__)
CORS(app) 

def get_model(obj_name):
    bucket_name = 'model-test-ng'
    try:
        # Retrieve AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")  # Default to us-east-1 if not set

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_default_region
        )

        # Retrieve the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=obj_name)

        # Get the file content as bytes
        file_content = response['Body'].read()

        # Load the model from the file content
        model = pickle.loads(file_content)

        print("Model loaded successfully.")
        return model
    
    except NoCredentialsError:
        print("Credentials not available.")
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
    except ClientError as e:
        print(f"Client error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

def run_average_model():
    model = get_model("average_demand_trained_model_v1.pkl")

    columns = ['Weekday','Season_Fall','Season_Spring','Season_Summer','Season_Winter',
       'Size', 'Fraction']
    data = {0:[0,0,0,0,1,1000,0.9]}

    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)

    predictions = model.predict(df)
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