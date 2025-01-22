from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
with open("model/house_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
    
with open("model/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    location = label_encoder.transform([data["location"]])[0]
    area = float(data["area"])
    bedrooms = int(data["bedrooms"])
    baths = int(data["baths"])
    
    # Prepare input for prediction
    input_data = np.array([[location, area, bedrooms, baths]])
    predicted_price = model.predict(input_data)[0]
    
    return jsonify({"predicted_price": f"{predicted_price:,.2f} PKR"})

if __name__ == "__main__":
    app.run(debug=True)
