import re
from typing import List

from fastapi import HTTPException
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate


class PopularTextProcessor:
    def __init__(self):
        self.llm = Ollama(model="llama3.1")

    async def process_text(self, text: str) -> List[str]:
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
        Если текст подходит под несколько категорий, выводи в формате "название категории / название категории".
        Если текст не подходит под категории отвечай "другое"."""),
            ("human", "Текст для классификации: {text}"),
        ])

        chain = prompt | self.llm
        try:
            response = await chain.invoke({"text": text})
            raw_text = response.generations[0][0].text
            cleaned_text = self._clean_text(raw_text)
            result = cleaned_text.split('/')
            return [item.strip() for item in result if item.strip()]

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[a-zA-Z]', '', text)
        print(text.strip())

        return text.strip()


# processor = PopularTextProcessor()
# # result = await processor.process_text("большая зеленая утка")
#
#
# # @app.post("/process-text/")
# async def process_text():
#     result = await processor.process_text("большая зеленая утка")
#     print(result)
# process_text()

llm = Ollama(model="llama3.1")

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
Если текст подходит под несколько категорий, выводи в формате "название категории / название категории".
Если текст не подходит под категории отвечай "другое"."""),
    ("human", "Текст для классификации: {text}"),
])

chain = prompt | llm

# Примеры текстов для классификации
texts = [
    "женщина в меховой шубе",
    "морской конек в воде",
    "Ученые открыли новый вид динозавров",
    "Футбольный матч между Барселоной и Реалом завершился со счетом 2:1"
]

for text in texts:
    response = chain.invoke({"text": text})
    print(response)
    # print(f"Текст: {text}")
    # print(f"Категория: {response.strip()}")
    # print("-" * 50)