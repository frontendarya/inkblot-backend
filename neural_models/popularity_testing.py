import re
from evaluate import load
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import numpy as np

# 1. Инициализация модели
llm = Ollama(model="llama3.1")


# 2. Очистка текста от символов, латиницы и цифр
def _clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    return text.strip()


# 3. Промпт
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты - классификатор текстов для теста Роршаха. Выбери строго одну наиболее подходящую категорию из списка:
летучая мышь;
бабочка;
жук;
любое четвероногое в обычном или боковом положении;
два человека;
галстук-бабочка;
человек или человекоподобное существо с поднятыми руками;
передняя часть насекомого;
шкура, меховая одежда, меховой ковер;
головы или лица женщин;
голова животного;
голова зайца;
млекопитающее;
морской конек;
многоногое животное;
насекомое;
краб;

Твой ответ должен содержать ТОЛЬКО номер и название категории в формате "название категории", ничего больше.
Если текст подходит под несколько категорий, выводи в формате "название категория / название категория".
Если текст не подходит под категории — отвечай "другое"."""),

    ("human", "Текст для классификации: {text}"),
])

# 4. Список тестовых текстов и эталонных ответов
test_data = [
    ("большая фиолетовая бабочка", ["бабочка"]),
    ("что-то похожее на жука", ["жук"]),
    ("два человека обнимаются", ["два человека"]),
    ("голова зайца", ["голова зайца"]),
    ("красивый меховой ковер", ["шкура, меховая одежда, меховой ковер"]),
    ("странное насекомое", ["насекомое"]),
    ("четвероногое животное на боку", ["любое четвероногое в обычном или боковом положении"]),
    ("голова женщины", ["головы или лица женщин"]),
    ("что-то похожее на краба", ["краб"]),
    ("непонятное пятно", ["другое"]),
]

exact_match_metric = load("exact_match")

scores = []

for i, (text, reference) in enumerate(test_data):
    chain = prompt | llm
    response = chain.invoke({"text": text})

    print(f"\n[{i + 1}] Вход: {text}")
    print(f"Ответ модели: {response}")

    cleaned = _clean_text(response)
    prediction = cleaned.strip()
    reference_str = reference[0].strip()

    print(f"Предсказание: {prediction}")
    print(f"Ожидается: {reference}")

    result = exact_match_metric.compute(predictions=[prediction], references=[reference_str])
    print(f"Точное совпадение: {result['exact_match']}")
    scores.append(result["exact_match"])

average_score = np.mean(scores)
print(f"\nСреднее значение Exact Match по 10 примерам: {average_score:.2f}")
