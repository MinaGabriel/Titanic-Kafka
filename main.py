import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available. GPU is ready for use.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU instead.")


import json

from kafka import KafkaProducer

# Create a Kafka producer instance
# 'bootstrap_servers' should point to the Kafka broker(s)
producer = KafkaProducer(bootstrap_servers='172.19.0.4:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Define the topic name
topic_name = 'my_topic'

# Create a JSON message
message = {
    'greeting': 'Hello',
    'target': 'mina'
}


producer.send(topic_name, message)

# Ensure all messages are sent and then close the producer
producer.flush()
producer.close()
