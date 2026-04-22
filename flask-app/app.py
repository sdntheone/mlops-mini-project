import os
import json
import pickle
import mlflow
import dagshub
from flask import Flask, render_template, request
from preprocessing_utility import normalize_text

os.environ['MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT'] = '1000'

mlflow.set_tracking_uri('https://dagshub.com/sdntheone/mlops-mini-project.mlflow')
dagshub.init(repo_owner='sdntheone', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

def load_mlflow_model():
    try:
        with open("reports/model_info.json", "r") as f:
            model_info = json.load(f)
        model_uri = model_info["model_uri"]
        print(f"[*] Attempting to fetch model from: {model_uri}")
        local_cache_path = os.path.join(os.getcwd(), "mlflow_model_cache")
        model = mlflow.pyfunc.load_model(model_uri, dst_path=local_cache_path)
        print("[+] Model loaded successfully from MLflow!")
        return model
    
    except Exception as e:
        print(f"[!] MLflow Load Error: {e}")
        print("[!] Falling back to local pickle file if available...")
        return pickle.load(open("models/model.pkl","rb")) if os.path.exists("models/model.pkl") else (_ for _ in ()).throw(Exception("Could not load model from MLflow OR local storage."))

model = load_mlflow_model()

try:
    vectoriser = pickle.load(open("models/vectorizer.pkl","rb"))
except FileNotFoundError:
    print("[!] Vectorizer not found at models/vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text','')
        if not text: return "No text provided", 400
        processed_text = normalize_text(text)
        features = vectoriser.transform([processed_text])
        result = model.predict(features)
        return render_template('index.html', result=result[0])
    except Exception as e:
        return f"Prediction Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)