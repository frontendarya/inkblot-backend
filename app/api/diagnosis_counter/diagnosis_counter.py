from typing import List

from app.api.diagnosis_counter.popular_enum import Popular
from app.schemas.diagnosis import DiagnosisCounterIn, Diagnosis
from neural_models.meaning_model import MeaningModel
from neural_models.popularity_model import PopularModel


class DiagnosisCounter:
    def __init__(self, diagnosis: DiagnosisCounterIn):
        super().__init__()
        self.diagnosis_list = diagnosis.diagnosis

        self.answers_counter = sum(
            [1 for item in self.diagnosis_list if item.is_passed is False])  # счётчик общего числа ответов
        self.rejection_counter = sum(
            [1 for item in self.diagnosis_list if item.is_passed is True])  # счётчик общего числа отказов

        # Чёткость формы
        self.scoreFPlus = sum([1 for item in self.diagnosis_list if item.form is True])  # чёткая форма
        self.scoreFMinus = sum([1 for item in self.diagnosis_list if item.form is False])  # размытая форма

        # Локализация
        self.scoreW = sum([1 for item in self.diagnosis_list if item.part == 1])  # целое изображение
        self.scoreD = sum([1 for item in self.diagnosis_list if item.part == 2])  # большие части пятна
        self.scoreDd = sum([1 for item in self.diagnosis_list if item.part == 3])  # малые детали пятна
        self.scoreS = sum([1 for item in self.diagnosis_list if item.part == 4])  # белый фон

        # Цвет и форма
        self.scoreFC = sum([1 for item in self.diagnosis_list if item.form_or_color == 1])  # форма доминирует
        self.scoreCF = sum([1 for item in self.diagnosis_list if item.form_or_color == 2])  # цвет доминирует
        self.scoreC = sum([1 for item in self.diagnosis_list if item.form_or_color == 3])  # цвет
        self.scoreF = sum([1 for item in self.diagnosis_list if item.form_or_color == 4])  # форма

    async def get_diagnosis(self):
        # Популярность
        scoreOrig = 0  # оригинальные ответы
        scorePopular = 0  # популярные ответы

        # Кинестетические показатели
        scoreM = 0  # движение/ искажение
        scoreMH = 0  # человеческие кинестезии
        scoreMA = 0  # кинестезии животных

        scoreH = 0  # СЧЁТЧИК ОТВЕТОВ человек
        scoreA = 0  # СЧЁТЧИК ОТВЕТОВ животное
        scoreMythH = 0  # магические человеки
        diagnosis_list = []

        popular_processor = PopularModel()
        model = MeaningModel()
        for item in self.diagnosis_list:
            if item.is_passed:
                card_id = item.card_id
                description = item.description

                popular_category = popular_processor.process_text(description)
                is_popular = self.count_popular(self, popular_category, card_id)
                if is_popular == 0:
                    scoreOrig += 1
                elif is_popular == 1:
                    scorePopular += 1
                else:
                    # TODO: логирование (ошибка в match/case)
                    raise ValueError(f"No popular category found with card_id {card_id}")

                prediction = await model.predict(description)
                if prediction in ['H', 'MH', 'A']:
                    # TODO: тоже отправить в модель
                    scoreMH += 1

        """
        посчитать количество по категориям
        отправить в модель ответы
        человек или нет (посчитать кинестезии)
        вызов функций 
        :return:
        """
        diagnosis_list.append(self.count_determinants(scoreMH, scoreMA, scorePopular, scoreOrig, scoreA, scoreH))
        diagnosis_list.append(self.count_type_of_experience(scoreM))
        diagnosis_list.append(self.count_intelligence(scoreA, scoreOrig, scoreM))
        diagnosis_list.append(self.count_conflict(scoreMythH, scoreM))
        return List[Diagnosis]

    # Подсчёт количества популярных ответов
    def count_popular(self, category, card_num):
        popular = 0
        category = Popular(category)
        if category is None:
            return popular
        # TODO добавить логирование (если не найдена категория - вызвать исключение)
        if category != Popular.OTHER:
            match card_num:
                case 1:
                    if category == (Popular.BAT or Popular.BUTTERFLY or Popular.BUG):
                        popular += 1
                case 2:
                    if category == Popular.QUADRUPED:
                        popular += 1
                case 3:
                    if category == (
                            Popular.TWO_PEOPLE or Popular.BUTTERFLY or Popular.BOW_TIE or Popular.HUMANOID or Popular.INSECT_FRONT_PART):
                        popular += 1
                case 4:
                    if category == Popular.FUR_SKIN:
                        popular += 1
                case 5:
                    if category == (Popular.BAT or Popular.BUTTERFLY):
                        popular += 1
                case 6:
                    if category == Popular.FUR_SKIN:
                        popular += 1
                case 7:
                    if category == (Popular.WOMAN_HEAD or Popular.ANIMAL_HEAD):
                        popular += 1
                case 8:
                    if category == Popular.MAMMAL:
                        popular += 1
                case 9:
                    if category == Popular.ANIMAL_HEAD:
                        popular += 1
                case 10:
                    if category == (Popular.HARE_HEAD or Popular.MULTI_LEGGED_ANIMAL or Popular.INSECT or Popular.CRAB):
                        popular += 1
                case _:
                    return 0
        return popular

    # Общие показатели
    def count_determinants(self, scoreMH, scoreMA, scorePopular, scoreOrig, scoreA, scoreH):
        determinants = []
        if (self.scoreDd / self.answers_counter) > 0.15:
            determinants.append('d')
        if (self.scoreS / self.answers_counter) > 0.4:
            determinants.append('S')
        if (self.scoreF / self.answers_counter) > 0.5:
            determinants.append('F')
        if scoreMH > 3:
            determinants.append('MH')
        if scoreMA == 0:
            determinants.append('MA')
        if (self.scoreC + self.scoreCF + self.scoreFC) == 0:
            determinants.append('C')
        if (scoreA / self.answers_counter) > 0.5:
            determinants.append('A')
        if (scoreH / self.answers_counter) > 0.15:
            determinants.append('H')
        if scorePopular == 0:
            determinants.append('Popular')
        if scoreOrig == 0:
            determinants.append('Orig-')
        elif scoreOrig > 0:
            determinants.append('Orig+')
        return determinants

    # Тип переживания
    def count_type_of_experience(self, scoreM):
        c = (3 * self.scoreC + 2 * self.scoreCF + self.scoreFC) / 2
        if (scoreM <= 1) & (c <= 1):
            return 'коартированный'
        elif (scoreM <= 3) & (c <= 3):
            return 'коартивный'
        elif (scoreM - c <= 3) & (scoreM <= 3) & (c <= 3):
            return 'aмбиэквальный'
        elif (scoreM - c) > 3:
            return 'интраверсивный'
        elif (c - scoreM) > 3:
            return 'экстратенсивный'
        else:
            return ''

    # Интеллект
    def count_intelligence(self, scoreA, scoreOrig, scoreM):
        if (self.scoreF >= 0 & scoreA & scoreOrig != 0 & scoreM != 0) or (self.scoreFPlus > 4):
            return 'высокий'
        elif scoreM == 0:
            return 'низкий'
        elif scoreM != 0 and self.scoreW > (scoreM * 2):
            return 'недостаточно'
        else:
            return ''

    # Конфликт и способ защиты
    def count_conflict(self, scoreMythH, scoreM):
        conflict_type = []
        if ((self.scoreCF + self.scoreC) > self.scoreFC) & (self.scoreF > 0.75):
            conflict_type += 'конфликт'
            if (self.rejection_counter > 0.4) & (scoreMythH != 0):
                conflict_type += 'вытеснение'
            elif scoreM:
                conflict_type += 'изоляция'
        else:
            return 'конфликт отсутствует'
        return conflict_type
