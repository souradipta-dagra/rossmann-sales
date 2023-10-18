from flask import Flask, render_template, request
from inference import inference_method
import pandas as pd

app = Flask(__name__)

@app.route('/health_check', methods = ['GET'])
def check():
    return "Yay! Flask App is running"
    
@app.route('/', methods = ['POST'])
def index():
    if request.method == 'POST':
        # try:
        print("Entered")
        store_data = pd.read_csv(request.files.get("store_data"))
        print("read the data")
        print(store_data.columns)
        df = pd.read_csv(request.files.get("test_data"))
        print(df.columns)
        predictions = inference_method(df, store_data)
        print("after predictions")
        return {"status_code":200,"message":"Sucess", "body": {"preds": list(predictions)}}
        # except Exception as e:

        #     return {"status_code":400,"message":"Failure", "body": {"msg": f"Error occured while model prediction == > {str(e)}"}}
    else:
        return {"status_code":400,"message":"Failure", "body": {"msg": "Request method not recognized"}}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)
