import re
from typing import List

# from fastapi import HTTPException
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate


# class PopularModel:
#     def __init__(self):
#         self.llm = Ollama(model="llama3.1")
#
#     async def process_text(self, text: str) -> List[str]:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """Ты - классификатор текстов для теста Роршаха. Выбери строго одну наиболее подходящую категорию из списка:
#         летучая мышь;
#         бабочка;
#         жук;
#         любое четвероногое в обычном или боковом положении;
#         два человека;
#         галстук-бабочка;
#         человек или человекоподобное существо с поднятыми руками;
#         передняя часть насекомого;
#         шкура, меховая одежда, меховой ковер;
#         головы или лица женщин;
#         голова животного;
#         голова зайца;
#         млекопитающее;
#         морской конек;
#         многоногое животное;
#         насекомое;
#         краб;
#
#         Твой ответ должен содержать ТОЛЬКО номер и название категории в формате "название категории", ничего больше.
#         Если текст подходит под несколько категорий, выводи в формате "название категории / название категории".
#         Если текст не подходит под категории отвечай "другое"."""),
#             ("human", "Текст для классификации: {text}"),
#         ])
#
#         chain = prompt | self.llm
#         try:
#             response = await chain.invoke({"text": text})
#             raw_text = response.generations[0][0].text
#             cleaned_text = self._clean_text(raw_text)
#             result = cleaned_text.split('/')
#             return [item.strip() for item in result if item.strip()]
#
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
#
#     def _clean_text(self, text: str) -> str:
#         text = re.sub(r'[^\w\s]', '', text)
#         text = re.sub(r'\d+', '', text)
#         text = re.sub(r'[a-zA-Z]', '', text)
#         print(text.strip())
#
#         return text.strip()
#
#
# popular_processor = PopularModel()
# popular_category = popular_processor.process_text("большая бабочка")
# print(popular_category)

llm = Ollama(model="llama3.1")

def _clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    # print(text.strip())

    return text.strip()

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
text = "большая фиолетовая бабочка"

chain = prompt | llm
response = chain.invoke({"text": text})
print(f"Ответ пользователя: {text}")
print(f"Результат, полученный от модели: {response}")
cleaned_text = _clean_text(response)
result = cleaned_text.split('/')
print(f"Обработанный ответ модели: {result}")
