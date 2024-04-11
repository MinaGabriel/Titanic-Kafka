from flask import Flask, request, render_template, jsonify
from titanic_nn import TitanicNN
import torch

app = Flask(__name__)

# Load the model
model = TitanicNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/infer', methods=['GET', 'POST'])
def infer():
    if request.method == 'POST':
        data = request.get_json()

        values = [
            data.get('Embarked', 0.0),
            data.get('Fare', 0.0) / 512,
            data.get('Parch', 0.0),
            data.get('Pclass', 0.0),
            data.get('Sex', 0.0),
            data.get('SibSp', 0.0),
            data.get('Relative', 0.0),
            data.get('AgeGroup', 0.0),
            data.get('Title', 0.0)
        ]

        tensor = torch.tensor([values], dtype=torch.float32)
        
        print(tensor)
        
        # Perform inference
        with torch.no_grad():  # We do not need to track gradients for inference
            predictions = model(tensor)
            print(predictions)

        return jsonify({'predictions': round(predictions.item(),2)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
