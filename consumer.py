from kafka import KafkaConsumer
import json
from backend.predict_model import predict  # real model

consumer = KafkaConsumer(
    'bank_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    data = message.value
    prediction, probability = predict(data)
    print(f"Received message: {data}")
    print(f'Predicted Class: {prediction}') 
    print(f'Probability of Conversion: {probability:.2f}')

# run consumer.py to simulate real time data streaming and give prediction