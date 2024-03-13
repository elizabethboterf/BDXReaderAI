import numpy as np
import torch
from joblib import load
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# load encoder, tokenizer and model
label_encoder = load('label_encoder.joblib')
tokenizer = DistilBertTokenizer.from_pretrained('tokenizer')
model = DistilBertForSequenceClassification.from_pretrained('distilBERT_model_3')

# The helper function to return the predictions and their probabilities from the model
def predict(columns):
    # Encode the columns
    inputs = tokenizer(columns, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

        # Convert to probabilities (optional)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

       # Get predicted classes
        predicted_classes = torch.argmax(probabilities, dim=1)
        # Convert class indices to labels
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
        # predicted_labels = [model.config.id2label[class_idx] for class_idx in predicted_classes.tolist()] 


        # Gather the probabilities of the predicted classes
        predicted_probabilities = probabilities[torch.arange(probabilities.size(0)), predicted_classes] 

        # Convert to list of tuples (class, probability)
        class_probability_pairs = list(zip(columns, predicted_labels, predicted_probabilities.tolist()))

        return class_probability_pairs


# The API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Get the columns from the POST request
    data = request.get_json(force=True)
    columns = data['columns']
    
    # Get predictions
    prediction = predict(columns)
    
    # Return the prediction
    return jsonify(prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
