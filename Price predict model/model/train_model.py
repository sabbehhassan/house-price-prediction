import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Handle missing values
data = data.dropna()

# Encode categorical columns
label_encoder = LabelEncoder()
data["Location"] = label_encoder.fit_transform(data["F-8"])

# Features and target
X = data[["Location", "26", "4", "5"]]
X.columns = ["Location", "Area", "Bedrooms", "Baths"]
y = data["420000"]
y.name = "Price"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("model/house_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the label encoder
with open("model/label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("Model and label encoder saved successfully!")
