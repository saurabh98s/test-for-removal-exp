import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('plant_growth_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the dataset into features and target
X = df.drop('Growth_Milestone', axis=1)
y = df['Growth_Milestone']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert numerical features into a text format for RoBERTa
def features_to_text(features):
    return " ".join(map(str, features))

train_texts = [features_to_text(sample) for sample in X_train]
test_texts = [features_to_text(sample) for sample in X_test]

# Labels remain the same
train_labels = y_train.tolist()
test_labels = y_test.tolist()

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from rexplain import removal, summary
from rexplain.behavior import PredictionGame
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y))).to(device)

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluation
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy}')

def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probabilities

def text_to_array(text):
    return np.array(tokenizer.encode(text, truncation=True, padding='max_length', max_length=128))

# Example text data for explanation
sample_text = "This is an example text for classification."
x = np.array([text_to_array(sample_text)])

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

        tokenized_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in x_]

        pred = predict_proba(tokenized_texts)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)

extension = MarginalExtension(x, predict_proba)

sample = np.array([text_to_array(sample_text)])
if sample.ndim == 1:
    sample = sample[np.newaxis]

game = PredictionGame(extension, sample)
attr = summary.ShapleyValue(game)
attr = attr.flatten()

plt.bar(np.arange(len(attr)), attr)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance using Shapley Values")
plt.show()
