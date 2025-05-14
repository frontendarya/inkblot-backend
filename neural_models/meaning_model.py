import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertForSequenceClassification, BertTokenizer


class MeaningModel:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('bert_finetuned')
        self.tokenizer = BertTokenizer.from_pretrained('bert_finetuned')

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        words = text.split()
        words = [word for word in words if word.isalpha()]
        return ' '.join(words)

    async def predict(self, text: str):
        processed_text = self.preprocess_text(text)

        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1)

        categories = ['H', '(H)', '(Hd)', 'А', '(А)', 'Ad', 'At', 'Sex', 'Obj', 'Aobj',
                      'Aat', 'Food', 'N', 'Geo', 'PI', 'Arch', 'Art', 'Abs', 'ВІ', 'Ті', 'Cl']

        mlb = MultiLabelBinarizer(classes=categories)

        # Создание словаря {индекс: название_категории}
        category_index_to_name = {i: cat for i, cat in enumerate(mlb.classes_)}
        predicted_categories = mlb.inverse_transform([pred.item()])[0]

        return pred.item()