import os

from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

iris = load_iris()
X_train = iris.data
y_train = iris.target

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

@app.route('/')
def home():
    api_key = os.environ.get("API_KEY", "no-api-key-found")# Dodanie obsługi zmiennej środowiskowej
    return jsonify({"message": "Witaj w API z modelem Iris!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Brak danych wejściowych! Podaj 'features' jako listę 4 liczb w JSON."}), 400

    try:
        input_data = np.array(data['features']).reshape(1, -1)
        if input_data.shape[1] != 4:
            return jsonify({"error": "Wymagane 4 cechy (sepal length, sepal width, petal length, petal width)!"}), 400

        prediction = model.predict(input_data)
        class_name = iris.target_names[prediction[0]]
        return jsonify({
            "prediction": int(prediction[0]),
            "class_name": class_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model_type": "LogisticRegression",
        "number_of_features": X_train.shape[1],
        "target_classes": iris.target_names.tolist(),
        "description": "Model klasyfikujący gatunki irysów na podstawie 4 cech."
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) # Zmiana z potrtu 5000 poniewarz Cloud nasłuchuje na 8080

