from transformers import BertTokenizer, BertForSequenceClassification
from rexplain import removal, summary
from rexplain.behavior import PredictionGame
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Check if CUDA is available and use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory before running the code
clear_gpu_memory()

# Initialize the tokenizer and model for BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Use DataParallel to use both GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

# Define the prediction function
def predict_proba(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    clear_gpu_memory()  # Clear GPU memory after predictions
    return probabilities
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Construct the prompt string separately
prompt = "\n".join([f'{message["role"]}: {message["content"]}' for message in messages])

# Adapt the input processing for text data
def text_to_array(text):
    return np.array(tokenizer.encode(text, truncation=True, padding='max_length', max_length=128))

# Example text data
texts = ["You are a pirate chatbot who always responds in pirate speak!", "Who are you?"]
x = np.array([text_to_array(text) for text in texts])

# Ensure predict_proba returns probabilities as expected
y = predict_proba(texts)

clear_gpu_memory()

# 1) Feature removal
class MarginalExtension:
    def __init__(self, data, model):
        self.model = model
        self.data = data
        self.data_repeat = data
        self.samples = len(data)

    def __call__(self, x, S):
        n = len(x)
        x = np.repeat(x, self.samples, axis=0)
        S = np.repeat(S, self.samples, axis=0)

        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        x_ = x.copy()
        for i in range(x_.shape[0]):
            x_[i, ~S[i]] = self.data_repeat[i, ~S[i]]

        # Convert the token IDs back to strings
        tokenized_texts = [tokenizer.decode(ids) for ids in x_]

        pred = predict_proba(tokenized_texts)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)

extension = MarginalExtension(x, predict_proba)

# 2) Model behavior
sample = np.array([text_to_array(prompt)])
if sample.ndim == 1:
    sample = sample[np.newaxis]

print("x shape:", x.shape)
print("sample shape:", sample.shape)
print("sample content:", sample)

game = PredictionGame(extension, sample)

# 3) Summary technique
attr = summary.ShapleyValue(game)

# Plotting the feature importances
plt.bar(np.arange(len(attr)), attr)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance using Shapley Values")
plt.show()
