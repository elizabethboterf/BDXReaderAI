import json
import numpy as np
import re
import torch
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

"""
Model Build proces
1. Gather and Clean Data, segregating into features and field_names
2. Encode field_names, 
3. Split Datasets, into test and training sets
4. Embed Text Features
5. Noramlize any Numerical Features
6. Create Dataset object with PyTorch
7. Create Training Arguments and Train the Model
8. Iteratively Evaluate and Fine-tune your Model
"""

# data colection and cleaning functions
def clean_text(text): 
    text = re.sub(r"[.,!?;:(){}[\]\"-]", " ", text)
    text = re.sub(r"[@#$%^&*|\\/+<>=`~]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()

    return text

column_names = []
field_names = []


# 1. Gather and Clean Data
with open(r'json_data/claims_fields_with_descriptions.json', 'r') as file:
    data = json.load(file)

    for fieldname, examples_array in data.items():
        column_and_description=""
        for i, obj in enumerate(examples_array):
            if(i<10):
                column_and_description= obj['column_name'] + ", " + obj['description']
            elif(i % 4 ==0): # we want to train on data that both with and without descriptions
                column_and_description= obj['column_name']

            column_names.append(column_and_description)
            field_names.append(fieldname)

print("done gathering")

# 2. Encode field_names
# load encoded field_names
encoder = LabelEncoder()
encoded_targets = encoder.fit_transform(field_names)
# Save the fitted label encoder to a file using joblib
dump(encoder, 'label_encoder.joblib')


# 3. Split Datasets
X_train, X_test, y_train, y_test = train_test_split(column_names, encoded_targets, test_size=0.2, random_state=42)

# Additional Step- I want to make sure all the test data is without descriptions, because that is how the real world situations will occur
for i in range(len(X_test)):
    X_test[i] = X_test[i].split(',', 1)[0]

# 4. Embed and Normalize Features
# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# embed the features
train_embeddings = tokenizer(X_train, truncation=True, padding=True)
test_embeddings = tokenizer(X_test, truncation=True, padding=True)

# 5. Save the Tokens and tokenizer
torch.save(train_embeddings, r'datasets/train_bert_tokens.pt')
torch.save(test_embeddings, r'datasets/test_bert_tokens.pt')
torch.save(y_train, r'datasets/train_bert_targets.pt')
torch.save(y_test, r'datasets/test_bert_targets.pt')

tokenizer.save_pretrained(r'tokenizer')

# 6. Create Dataset object with PyTorch on train file- next file

