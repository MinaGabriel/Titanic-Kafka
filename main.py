from flask import Flask, request, jsonify
import torch
from titanic_nn import TitanicNN  # Assuming your model class is defined in model_file.py

app = Flask(__name__)

# Hyperparameters
input_size = 9  # Assuming you have 9 features in your dataset
hidden_size = 64
output_size = 1  # Assuming you're predicting survival (binary classification)

# Load the saved model
model = TitanicNN(input_size, hidden_size, output_size)  # Initialize the model
model.load_state_dict(torch.load('model.pth'))  # Load the model weights

# Define endpoint for model inference
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json  # Assuming input data is sent as JSON

    # Convert the input data to a PyTorch tensor
    input_data = torch.tensor([list(data.values())], dtype=torch.float32)

    # Set the model to evaluation mode
    model.eval()

    # Pass input data through the model to get predictions
    with torch.no_grad():
        predictions = model(input_data)

    # Convert predictions to probabilities (if needed)
    probabilities = torch.sigmoid(predictions)  # Assuming the model output is logits for binary classification

    # Prepare the response
    response = {
        'prediction': probabilities.item()  # Convert tensor to Python float
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
