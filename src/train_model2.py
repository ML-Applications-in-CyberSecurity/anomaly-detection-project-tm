import os
import json
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COMMON_PORTS = [80, 443, 22, 8080]

def generate_normal_data():
    return {
        "src_port": random.choice(COMMON_PORTS),
        "dst_port": random.randint(1024, 65535),
        "packet_size": random.randint(100, 1500),
        "duration_ms": random.randint(50, 500),
        "protocol": random.choice(["TCP", "UDP"])
    }

dataset = [generate_normal_data() for _ in range(1000)]

os.makedirs('./dataset', exist_ok=True)

data_file_path = './dataset/training_data.json'
with open(data_file_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Data has been successfully saved to {data_file_path}")

with open(data_file_path) as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
print("Dataset loaded into DataFrame:")
print(df.head())

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['protocol'])
    return np.array(df)

processed_data = preprocess_data(df)

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(processed_data)


script_dir = os.path.dirname(os.path.abspath(__file__))  
model_file_path = os.path.join(script_dir, "anomaly_model.joblib")
# model_file_path = "./anomaly_model.joblib"
joblib.dump(model, model_file_path)

print(f"Model has been successfully trained and saved to {model_file_path}")


predictions = model.predict(processed_data)
print(f"Predictions made: {predictions[:10]}")  

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(processed_data)

normal_data = reduced_data[predictions == 1]
anomalous_data = reduced_data[predictions == -1]

plt.scatter(normal_data[:, 0], normal_data[:, 1], color='green', label='Normal')
plt.scatter(anomalous_data[:, 0], anomalous_data[:, 1], color='red', label='Anomalous')
plt.legend()
plt.title('PCA of Normal and Anomalous Data')
plt.show()
