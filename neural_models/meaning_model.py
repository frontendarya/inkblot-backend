import nltk
import pandas as pd
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

morph = MorphAnalyzer()

df = pd.read_csv("./data/data.csv", encoding='utf-8').fillna('')
for column in df.columns:
    df[column] = df[column].replace(1.0, column)
df = shuffle(df)

# Предобработка текста
def preprocess_text(text):
    text = text.lower()  # Приведение к нижнему регистру
    words = text.split()  # Токенизация
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Удаление стоп-слов и пунктуации
    words = [morph.normal_forms(word)[0] for word in words]  # Лемматизация
    return ' '.join(words)

df['title'] = df['title'].apply(preprocess_text)

labels = df.drop('title', axis=1).unique().to_list()
# df['tags'] = df[labels].apply(
#     lambda x: [tag for tag in labels if x[tag] == tag],
#     axis=1
# )

# Кодируем метки
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['tags'])
# X_train, X_test, y_train, y_test = train_test_split(df['title'], df.drop('title', axis=1), test_size=0.2, random_state=42)
#
# encoder = LabelEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.transform(y_test)
#
#


print(df.head())

# from transformers import BertTokenizer
# from torch.utils.data import Dataset
# import torch
#
#
# class MultiLabelDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, item):
#         text = str(self.texts[item])
#         label = self.labels[item]
#
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#
#         return {
#             'text': text,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.FloatTensor(label)
#         }
#
#
# from transformers import BertModel
# import torch.nn as nn
#
#
# class BertMultiLabelClassifier(nn.Module):
#     def __init__(self, model_name, num_labels):
#         super(BertMultiLabelClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         return self.classifier(pooled_output)
#
#
# from transformers import AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader
# import numpy as np
# from sklearn.metrics import f1_score, accuracy_score
#
#
# def train_model(model, train_data, val_data, learning_rate, epochs, batch_size):
#     train_dataset = MultiLabelDataset(
#         train_data['texts'],
#         train_data['labels'],
#         tokenizer,
#         max_len=128
#     )
#
#     val_dataset = MultiLabelDataset(
#         val_data['texts'],
#         val_data['labels'],
#         tokenizer,
#         max_len=128
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     total_steps = len(train_loader) * epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=0,
#         num_training_steps=total_steps
#     )
#
#     loss_fn = nn.BCEWithLogitsLoss().to(device)
#
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#
#         for batch in train_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#
#             model.zero_grad()
#             outputs = model(input_ids, attention_mask)
#             loss = loss_fn(outputs, labels)
#             total_loss += loss.item()
#
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()
#
#         avg_train_loss = total_loss / len(train_loader)
#
#         # Валидация
#         model.eval()
#         val_loss = 0
#         predictions, true_labels = [], []
#
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['labels'].to(device)
#
#                 outputs = model(input_ids, attention_mask)
#                 loss = loss_fn(outputs, labels)
#                 val_loss += loss.item()
#
#                 preds = torch.sigmoid(outputs).cpu().detach().numpy()
#                 predictions.append(preds)
#                 true_labels.append(labels.cpu().detach().numpy())
#
#         avg_val_loss = val_loss / len(val_loader)
#         predictions = np.concatenate(predictions, axis=0)
#         true_labels = np.concatenate(true_labels, axis=0)
#
#         # Порог для классификации
#         threshold = 0.5
#         pred_labels = (predictions > threshold).astype(int)
#
#         # Метрики
#         f1 = f1_score(true_labels, pred_labels, average='micro')
#         accuracy = accuracy_score(true_labels, pred_labels)
#
#         print(f'Epoch {epoch + 1}/{epochs}')
#         print(f'Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')
#         print(f'Val F1 (micro): {f1:.4f}, Val Accuracy: {accuracy:.4f}')
#         print('-' * 50)
#
#
# # Параметры
# model_name = 'bert-base-multilingual-cased'
# num_labels = len(label_columns)  # количество классов
# batch_size = 16
# epochs = 3
# learning_rate = 2e-5
#
# # Инициализация
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertMultiLabelClassifier(model_name, num_labels).to(device)
#
# # Обучение
# train_model(
#     model,
#     train_data,
#     val_data,
#     learning_rate,
#     epochs,
#     batch_size
# )
#
# from transformers import AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader
# import numpy as np
# from sklearn.metrics import f1_score, accuracy_score
#
#
# def train_model(model, train_data, val_data, learning_rate, epochs, batch_size):
#     train_dataset = MultiLabelDataset(
#         train_data['texts'],
#         train_data['labels'],
#         tokenizer,
#         max_len=128
#     )
#
#     val_dataset = MultiLabelDataset(
#         val_data['texts'],
#         val_data['labels'],
#         tokenizer,
#         max_len=128
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     total_steps = len(train_loader) * epochs
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=0,
#         num_training_steps=total_steps
#     )
#
#     loss_fn = nn.BCEWithLogitsLoss().to(device)
#
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#
#         for batch in train_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#
#             model.zero_grad()
#             outputs = model(input_ids, attention_mask)
#             loss = loss_fn(outputs, labels)
#             total_loss += loss.item()
#
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             scheduler.step()
#
#         avg_train_loss = total_loss / len(train_loader)
#
#         # Валидация
#         model.eval()
#         val_loss = 0
#         predictions, true_labels = [], []
#
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['labels'].to(device)
#
#                 outputs = model(input_ids, attention_mask)
#                 loss = loss_fn(outputs, labels)
#                 val_loss += loss.item()
#
#                 preds = torch.sigmoid(outputs).cpu().detach().numpy()
#                 predictions.append(preds)
#                 true_labels.append(labels.cpu().detach().numpy())
#
#         avg_val_loss = val_loss / len(val_loader)
#         predictions = np.concatenate(predictions, axis=0)
#         true_labels = np.concatenate(true_labels, axis=0)
#
#         # Порог для классификации
#         threshold = 0.5
#         pred_labels = (predictions > threshold).astype(int)
#
#         # Метрики
#         f1 = f1_score(true_labels, pred_labels, average='micro')
#         accuracy = accuracy_score(true_labels, pred_labels)
#
#         print(f'Epoch {epoch + 1}/{epochs}')
#         print(f'Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')
#         print(f'Val F1 (micro): {f1:.4f}, Val Accuracy: {accuracy:.4f}')
#         print('-' * 50)
#
#
# # Параметры
# model_name = 'bert-base-multilingual-cased'
# num_labels = len(label_columns)  # количество классов
# batch_size = 16
# epochs = 3
# learning_rate = 2e-5
#
# # Инициализация
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertMultiLabelClassifier(model_name, num_labels).to(device)
#
# # Обучение
# train_model(
#     model,
#     train_data,
#     val_data,
#     learning_rate,
#     epochs,
#     batch_size
# )
#
# # Сохранение
# model.save_pretrained('./saved_model')
# tokenizer.save_pretrained('./saved_model')
#
# # Загрузка
# from transformers import BertModel
#
# model = BertMultiLabelClassifier.from_pretrained('./saved_model')
# tokenizer = BertTokenizer.from_pretrained('./saved_model')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Инициализация BERT модели и токенизатора
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # Функция для векторизации текста с использованием BERT
# def encode_text(texts, tokenizer, max_length=64):
#     inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Используем эмбеддинг [CLS] токена (первый токен)
#     return outputs.last_hidden_state[:, 0, :].numpy()
#
# # Преобразуем текст в эмбеддинги
# X = encode_text(df['title'].tolist(), tokenizer)
#
# # Преобразуем метки в массив numpy
# y = df.drop('title', axis=1).values
#
# # Разделяем на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Преобразуем в тензоры PyTorch
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)
#
# # Создаем Dataset и DataLoader для PyTorch
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
#
# # Пример использования DataLoader
# for data, target in train_loader:
#     print(f"Features: {data.shape}, Target: {target.shape}")
#
#
# class BertWithAttention(nn.Module):
#     def __init__(self, num_classes=21):
#         super(BertWithAttention, self).__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.attention = nn.Sequential(
#             nn.Linear(self.bert.config.hidden_size, 128),
#             nn.Tanh(),
#             nn.Linear(56, 1),
#             nn.Softmax(dim=1)
#         )
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
#
#         # Attention mechanism
#         attn_weights = self.attention(hidden_states)  # (batch_size, seq_len, 1)
#         context = torch.sum(attn_weights * hidden_states, dim=1)  # (batch_size, hidden_size)
#
#         return self.classifier(context)
#
# model = BertWithAttention()
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
#
# for batch in train_loader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     labels = batch['labels']
#
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     print(outputs.shape)  # [batch_size, num_classes]
#     break


#
# import multiprocessing
# from gensim.models import Word2Vec
# import nltk
# from nltk.tokenize import word_tokenize
#
# cores = multiprocessing.cpu_count()
# nltk.download('punkt_tab')
#
# w2v_model = Word2Vec(sentences,
#                      min_count=10,
#                      window=4,
#                      sample=6e-5,
#                      alpha=0.03,
#                      min_alpha=0.0003,
#                      negative=20,
#                      workers=cores-1)
# w2v_model.save("word2vec.model")
#
# def text_to_embedding(text, model):
#     words = text.split()
#     word_vectors = []
#     for word in words:
#         if word in model:
#             word_vectors.append(model[word])
#     if word_vectors:
#         return np.mean(word_vectors, axis=0).tolist()
#     else:
#         try:
#             return [0.0] * model.vector_size * 2    # Если слова нет, то вернём нулевой вектор (по аналогии с CountVectorizer)
#         except AttributeError:
#             return [0.0] * 300  # navec и rusvectors не имеют атрибута vector_size, зато длина их векторов известна и равна 300
