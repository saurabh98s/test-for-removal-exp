from transformers import BertTokenizer, BertForSequenceClassification
from rexplain import removal, summary
from rexplain.behavior import PredictionGame
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc

def clear_mps_memory():
    gc.collect()
    torch.mps.empty_cache()

# Check if MPS is available and use it if possible
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Clear MPS memory before running the code
clear_mps_memory()

# Initialize the tokenizer and model for BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name).to(device)

# Define the initial messages for the chatbot
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Construct the prompt string separately
prompt = "\n".join([f'{message["role"]}: {message["content"]}' for message in messages])

# Define the prediction function
def predict_proba(texts):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    clear_mps_memory()  # Clear MPS memory after predictions
    return probabilities

# Adapt the input processing for text data
def text_to_array(text):
    return np.array(tokenizer.encode(text, truncation=True, padding='max_length', max_length=128))

# Example text data
texts = ["You are a pirate chatbot who always responds in pirate speak!", "Who are you?"]
x = np.array([text_to_array(text) for text in texts])

# Ensure predict_proba returns probabilities as expected
y = predict_proba(texts)

clear_mps_memory()

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
        tokenized_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in x_]

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

clear_mps_memory()  # Final memory clearance
