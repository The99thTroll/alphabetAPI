from flask import Flask, jsonify, request
from main import getPrediction

app = Flask(__name__)

@app.route("/predict-data", methods = ["POST"])

def predictData():
    images = request.files.get("letter")
    prediction = getPrediction(images)
    
    return jsonify({
        'result': prediction
    }), 200
    
if (__name__ == "__main__"): 
    app.run(debug=True)