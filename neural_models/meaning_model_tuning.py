import re

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

"""
Предобработка формата датасета
try:
    df = pd.read_csv("./data/data.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("./data/data.csv", encoding='utf-8-sig')
df.columns = df.columns.str.strip()

categories = ['H', '(H)', '(Hd)', 'А', '(А)', 'Ad', 'At', 'Sex', 'Obj', 'Aobj',
              'Aat', 'Food', 'N', 'Geo', 'PI', 'Arch', 'Art', 'Abs', 'ВІ', 'Ті', 'Cl']

existing_categories = [cat for cat in categories if cat in df.columns]


def get_categories(row):
    return ', '.join([cat for cat in existing_categories if row[cat] == 1])


df['category'] = df.apply(get_categories, axis=1)

result = df[['title', 'category']]
result.to_csv("./data/result.csv", index=False, encoding='utf-8')
"""

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

morph = MorphAnalyzer()

df = pd.read_csv("./data/result.csv", encoding='utf-8')

for column in df.columns:
    df[column] = df[column].replace(1.0, column)
df = shuffle(df)


def clean_text(text):
    return re.sub(r'""', '', text)


def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    words = text.split()  # Токенизация
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Удаление стоп-слов и пунктуации
    words = [morph.normal_forms(word)[0] for word in words]  # Лемматизация
    return ' '.join(words)


df['title'] = df['title'].apply(preprocess_text)
df['category'] = df['category'].apply(clean_text)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['category'].apply(lambda x: x.split(', ') if isinstance(x, str) else []))
X_train, X_test, y_train, y_test = train_test_split(df['title'], y, test_size=0.2, random_state=42)


class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification"
)

train_dataset = BertDataset(X_train.values, y_train, tokenizer, MAX_LEN)
test_dataset = BertDataset(X_test.values, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}')

from sklearn.metrics import classification_report
import numpy as np

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.sigmoid(logits) > 0.5

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

print(classification_report(
    np.array(true_labels),
    np.array(predictions),
    target_names=mlb.classes_
))

model.save_pretrained('bert_finetuned')
tokenizer.save_pretrained('bert_finetuned')
