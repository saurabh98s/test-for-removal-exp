# Install necessary packages
# !pip install transformers torch rexplain

# Import libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from rexplain import removal, behavior, summary
import matplotlib.pyplot as plt

# Load the tokenizer and model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input data
texts = ["I love this product!", "This is the worst service ever."]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Generate predictions
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
print(predictions)

# Define a function to predict sentiment using the LLM
def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits.numpy()

# Convert texts to a format suitable for the explainer
encoded_texts = [tokenizer.encode(t, add_special_tokens=True, truncation=True, padding='max_length', max_length=50) for t in texts]
x = np.array(encoded_texts)

# 1) Feature removal
extension = removal.MarginalExtension(x[:512], predict_sentiment)

# 2) Model behavior
game = behavior.PredictionGame(x[0], extension)

# 3) Summary technique
attr = summary.ShapleyValue(game)

# Visualize the explanation
plt.bar(np.arange(len(attr)), attr)
plt.show()
