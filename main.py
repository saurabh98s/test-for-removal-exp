from transformers import pipeline
from rexplain import removal, behavior, summary
import numpy as np
import matplotlib.pyplot as plt
import torch

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Check if CUDA is available and use GPU if possible
device = 0 if torch.cuda.is_available() else -1

# Clear GPU memory before running the code
clear_gpu_memory()
# Define the initial messages for the pirate chatbot
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Initialize the text-generation pipeline with the specified model
chatbot = pipeline("text-generation", model="mistralai/Mistral-Nemo-Instruct-2407", max_new_tokens=128,device=0)

# Define the prediction function
def predict_proba(messages):
    # Flatten the messages into a single string prompt
    prompt = "\n".join([f'{message["role"]}: {message["content"]}' for message in messages])
    
    # Generate the response using the chatbot
    response = chatbot(prompt)
    
    # Extract the generated text
    generated_text = response[0]['generated_text']
    print(generated_text)
    return generated_text

# Convert messages to the format needed by rexplain
x = np.array([messages])
y = predict_proba(messages)

clear_gpu_memory()

# 1) Feature removal
extension = removal.MarginalExtension(x[:512], predict_proba)

# 2) Model behavior
game = behavior.PredictionGame(x[0], extension)

# 3) Summary technique
attr = summary.ShapleyValue(game)
plt.bar(np.arange(len(attr)), attr)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance using Shapley Values")
plt.show()
