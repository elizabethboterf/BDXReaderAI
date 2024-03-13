import torch
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, TrainerCallback, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from Dataset import CustomDataset
import json
import os

"""
Model Build proces
1. Gather and Clean Data, segregating into features and targets
2. Encode Targets, 
3. Split Datasets, into test and training sets
4. Embed Text Features
5. Noramlize any Numerical Features
6. Create Dataset object with PyTorch
7. Create Training Arguments and Train the Model
8. Iteratively Evaluate and Fine-tune your Model
"""

# 6. Create Dataset object with PyTorch on train file
# define the custom dataset class to format the data for the trainer


# load saved embeddings
train_embeddings = torch.load(r'datasets/train_bert_tokens.pt')
# val_embeddings =torch.load(r'datasets/val_bert_tokens.pt')
test_embeddings = torch.load(r'datasets/test_bert_tokens.pt')
# load saved targets
train_targets = torch.load(r'datasets/train_bert_targets.pt')
# val_targets = torch.load(r'datasets/val_bert_targets.pt')
test_targets =torch.load(r'datasets/test_bert_targets.pt')

# make the dataset
train_dataset = CustomDataset(train_embeddings, train_targets)
# val_dataset = Dataset(val_embeddings, val_targets)
test_dataset = CustomDataset(test_embeddings, test_targets)


# 7. Create Training Arguments and Train the Model
# Load DistilBERT model for sequence classification with the number of labels

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=14)

class MetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Save metrics to a file or database
        with open(r"distilBERT_model_1/metrics_log.txt", "a") as f:
            print(metrics, file=f)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
   # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute precision, recall, and F1-score with macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    
    # Compute precision, recall, and F1-score with weighted average
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    
    # Compute precision, recall, and F1-score with micro average
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, predictions, average="micro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro, "precision_macro": precision_macro, "recall_macro": recall_macro,
        "f1_weighted": f1_weighted, "precision_weighted": precision_weighted, "recall_weighted": recall_weighted,
        "f1_micro": f1_micro, "precision_micro": precision_micro, "recall_micro": recall_micro,
    }

training_args = TrainingArguments(
    output_dir='./results',          # output directory for model predictions and checkpoints
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=200,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    learning_rate=5e-5,
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,                 # Evaluate the model every 100 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    # callbacks=[MetricsLogger()]
)

trainer.train()

# 8. Iteratively Evaluate and Fine-tune your Model
# save the model
trainer.save_model(r'distilBERT_model_3')

# print the results
results = trainer.evaluate()
metrics_str = json.dumps(results, indent=4)

print(results)


# documenting the results
def save_to_excel(model, report, id):
    # Extract model configuration. DistilBERT models store configuration in the config attribute
    config = model.config.to_dict()
    print(config)

    # Define the parameters or configuration options you're interested in
    # Adjust this list based on the specifics of your model's configuration
    interested_keys = [
        'architectures',
        'attention_dropout',
        'dropout',
        'hidden_dim',
        'initializer_range',
        'max_position_embeddings',
        'model_type',
        'n_heads',
        'n_layers',
        'vocab_size'
    ]

    # Extract the relevant parameters
    # ordered_params = [config.get(key, 'N/A') for key in interested_keys]
    # ordered_params.insert(0, f"DistilBERT_model_{id}")
    # Add placeholder for separation or additional metadata
    # ordered_params.append(" ")

    # Assuming 'report' is the dictionary returned by your compute_metrics function
    # ordered_scores = [
    #     report['eval_loss'],
    #     report['eval_accuracy'],
    #     report['eval_precision_macro'],
    #     report['eval_recall_macro'],
    #     report['eval_f1_macro'],
    #     report['eval_precision_weighted'],
    #     report['eval_recall_weighted'],
    #     report['eval_f1_weighted'],
    #     report['eval_precision_micro'],
    #     report['eval_recall_micro'],
    #     report['eval_f1_micro'],
    # ]

    # row_list = ordered_params + ordered_scores
    df = pd.DataFrame([config])

    file_path = 'Model Versioning distilBERT.xlsx'  # Adjust the file path as needed
    sheet_name = 'Best Performance Models'

    # Append results to an existing Excel file without overwriting it
    try:
        file_path = r'C:\Users\ElizabethBoterf\Documents\BDXReaderModelReports\Model Versioning distilBERT.xlsx'
        sheet_name = 'Best Performance Models'

        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay', ) as writer:
            if sheet_name in writer.sheets:
                startrow = writer.sheets[sheet_name].max_row
            else: 
                startrow = 0
            df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False if startrow > 0 else True)

    except FileNotFoundError:
        # If the file does not exist, write a new file
        df.to_excel(file_path, sheet_name=sheet_name, index=False)

    print("Finished documenting report.")

def save_report_txt(model, report, id):
    # Assuming the model has a method to get its configuration or parameters in a dictionary form
    model_params = model.config.to_dict()  # Adjust this line based on how you get your model's parameters

    json_params = json.dumps(model_params, indent=4)

    model_text = f"Model: distilBERT_v_{id}\n"
    params_text = f"Params: \n{json_params}\n\n"

    # Prepare data for the DataFrame
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, (float, int)):  # For overall metrics like accuracy
            row = {"Metric": label, "Value": metrics}
            rows.append(row)
        # else:  # For metrics by label
        #     for metric_name, metric_value in metrics.items():
        #         if metric_name != 'support':  # Assuming you don't want to include 'support' in the report
        #             row = {"Metric": f"{label} {metric_name}", "Value": metric_value}
        #             rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)

    report_text = df.to_string(index=False)

    full_text = model_text + params_text + "Metrics Report:\n" + report_text

    report_folder_path = r'C:\Users\ElizabethBoterf\Documents\BDXReaderModelReports'
    report_filename = f"best_distilBERT_model_{id}_report.txt"
    report_file_path = os.path.join(report_folder_path, report_filename)

    os.makedirs(report_folder_path, exist_ok=True)

    with open(report_file_path, 'w') as file:
        file.write(full_text)
    print("Finished saving report to", report_file_path)

save_to_excel(trainer.model, results, 3)
save_report_txt(trainer.model, results, 3)