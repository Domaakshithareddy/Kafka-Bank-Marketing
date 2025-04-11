from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Sample values for random generation
jobs = ['admin.', 'technician', 'services', 'management', 'unemployed']
maritals = ['married', 'single', 'divorced']
educations = ['primary', 'secondary', 'tertiary']
defaults = ['yes', 'no']
housings = ['yes', 'no']
loans = ['yes', 'no']
contacts = ['cellular', 'telephone']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcomes = ['unknown', 'failure', 'success']

def generate_customer_data():
    return {
        "age": random.randint(18, 70),
        "job": random.choice(jobs),
        "marital": random.choice(maritals),
        "education": random.choice(educations),
        "default": random.choice(defaults),
        "balance": random.randint(-2000, 50000),
        "housing": random.choice(housings),
        "loan": random.choice(loans),
        "contact": random.choice(contacts),
        "day": random.randint(1, 31),
        "month": random.choice(months),
        "duration": random.randint(0, 1000),
        "campaign": random.randint(1, 10),
        "pdays": random.randint(-1, 100),
        "previous": random.randint(0, 5),
        "poutcome": random.choice(poutcomes)
    }

while True:
    data = generate_customer_data()
    producer.send('bank_customers', value=data)
    print(f"Sending: {data}")
    time.sleep(1)
