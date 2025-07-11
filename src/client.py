import socket
import json
import pandas as pd
import joblib
import requests
import csv

model = joblib.load("./anomaly_model.joblib")

def pre_process_data(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['protocol'])
    
    expected_columns = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_TCP', 'protocol_UDP']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df.to_numpy() 



def send_to_together_api(data, template_type=1):
    url = "https://api.together.ai/v1/chat/completions"  
    headers = {
        "Authorization": "4f48e940bedbb0da9a4cd1d05bb5c82be6efccfdecc0b41daf29fb7ad69d5066",  
        "Content-Type": "application/json"
    }

    if template_type == 1:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that labels sensor anomalies."},
            {"role": "user", "content": f"Sensor reading: {data}\nDescribe the type of anomaly and suggest a possible cause."}
        ]
    elif template_type == 2:
        messages = [
            {"role": "system", "content": "You are an expert in anomaly detection."},
            {"role": "user", "content": f"Received sensor data: {data}. Can you identify any anomaly?"}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a specialist in network monitoring."},
            {"role": "user", "content": f"Data: {data}\nAnalyze and explain any potential network anomaly."}
        ]

    response = requests.post(url, json={"messages": messages}, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        return response_data  
    else:
        print(f"Error: {response.status_code}")
        return None

def save_anomaly_to_csv(data):
    with open('anomalies.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data['src_port'], data['dst_port'], data['packet_size'], data['duration_ms'], data['protocol']])

HOST = 'localhost'
PORT = 9999

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                df = pre_process_data(data)  
                prediction = model.predict(df)
                confidence_score = model.decision_function(df)

                if prediction == -1:  
                    print("\n🚨 Anomaly Detected!\n")
                    save_anomaly_to_csv(data)  
                    result = send_to_together_api(data)  
                    if result:
                        label = result.get('choices', [{}])[0].get('message', {}).get('content', 'No label found')
                        print(f"API Response: {label}")

               
                confidence_score = model.decision_function(df)  
                print(f"Confidence Score: {confidence_score}")

            except json.JSONDecodeError:
                print("Error decoding JSON.")
