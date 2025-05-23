from alembic import op
import sqlalchemy as sa

revision = 'a1b2c3d4e5f6'
down_revision = '300aa0f065e6'
branch_labels = None
depends_on = None

def upgrade():
    users_table = sa.table('diagnosis',
                           sa.Column('id', sa.Integer()),
                           sa.Column('category', sa.String()),
                           sa.Column('short_name', sa.String()),
                           sa.Column('description', sa.String())
                           )

    op.bulk_insert(users_table,
                   [
                       {'id': 1, 'category': 'Общая оценка личности', 'short_name': 'd', 'description': 'Наличие большого количества ответов на мелкие детали может указывать на излишний педантизм или симптом навязчивости.'},
                       {'id': 2, 'category': 'Общая оценка личности', 'short_name': 's', 'description': 'Наличие большого количества ответов на мелкие детали указывает на склонность к использованию проекции в качестве механизма защиты.'},
                       {'id': 3, 'category': 'Общая оценка личности', 'short_name': 'f', 'description': 'Наличие большого количества ответов с чёткой формой указывает на беспристрастность, объективность, также может свидетельствовать на сверхконтроль, недостаток спонтанности и предрасположенность к конфликтам.'},
                       {'id': 4, 'category': 'Общая оценка личности', 'short_name': 'mh', 'description': 'Наличие большого количества человеческих кинестезий - признак осознанной или хорошо контролируемой внутренней жизни, интраверсивность, зрелось внутреннего «Я», творческий интеллект, способность к эматии.'},
                       {'id': 5, 'category': 'Общая оценка личности', 'short_name': 'ma', 'description': 'Отсутствие кинестезий животных может указывать на подавление примитивных влечений, возможно в силу их неприемлемого содержания.'},
                       {'id': 6, 'category': 'Общая оценка личности', 'short_name': 'c', 'description': 'Отсутствие ответов с наличием цвета может указывать на недостаточную адаптивность, возможно, глубокую дисфорию, грусть, отсутствие доверия к себе, депрессию'},
                       {'id': 7, 'category': 'Общая оценка личности', 'short_name': 'a', 'description': 'Наличие большого количества ответов из категории "Животные" указывает на стереотипию и бедность интересов.'},
                       {'id': 8, 'category': 'Общая оценка личности', 'short_name': 'h', 'description': 'Наличие большого количества ответов из категории "Человек" указывает на наличие социальной чувствительности и эмпатии.'},
                       {'id': 9, 'category': 'Общая оценка личности', 'short_name': 'popular', 'description': 'Отсутствие популярных ответов может указывать на патологический негативизм, аутизим или нарушения адаптации.'},
                       {'id': 10, 'category': 'Общая оценка личности', 'short_name': 'orig', 'description': 'Отсутствие оригинальных ответов - признак дезорганизации мышления, потери контакта с реальностью.'},
                       {'id': 11, 'category': 'Тип переживания', 'short_name': 'коартированный', 'description': 'Коартированный - этот тип переживания характеризуется сильной эмоциональной напряженностью и стремлением к действию. Он связан с активной борьбой с препятствиями и преодолением трудностей. Часто характеризует сухих чопорных людей, склонных к поучениям, не обладающих ни оригинальностью мышления, ни живостью чувств, зато стойких и надежных. Наряду с нормой эти типы встречаются у депрессивных невротиков или скомпенсированных больных шизофренией.'},
                       {'id': 12, 'category': 'Тип переживания', 'short_name': 'коартивный', 'description': 'Коартивный - этот тип переживания связан с преодолением внутренних препятствий, таких как страхи, тревоги, сомнения. Он характеризуется внутренней борьбой и напряженностью. Часто характеризует сухих чопорных людей, склонных к поучениям, не обладающих ни оригинальностью мышления, ни живостью чувств, зато стойких и надежных. Наряду с нормой эти типы встречаются у депрессивных невротиков или скомпенсированных больных шизофренией.'},
                       {'id': 13, 'category': 'Тип переживания', 'short_name': 'aмбиэквальный', 'description': 'Амбиэквальный - этот тип переживания характеризуется одновременным присутствием противоположных эмоций или чувств. Например, радость и грусть, любовь и ненависть могут присутствовать одновременно. Человек может замыкаться в себе, восстанавливая социальный ресурс, а затем с новыми силами возвращается к активностям во внешнем мире.'},
                       {'id': 14, 'category': 'Тип переживания', 'short_name': 'интраверсивный', 'description': 'Интраверсивный - этот тип переживания связан с интроспекцией, самонаблюдением и самоанализом. Он характеризуется глубоким внутренним переживанием. Мотиваторами для них являются внутренние стимулы, требования внешнего мира им не важны. У этого типа развита склонность к воображению, которое также может являться защитным механизом при негативном воздействии внешней среды.'},
                       {'id': 15, 'category': 'Тип переживания', 'short_name': 'экстратенсивный', 'description': 'Экстратенсивный - этот тип переживания связан с внешним миром и внешними стимулами. Он характеризуется активной реакцией на внешние события и обстоятельства. Тип часто характеризуется открытой экспрессией, широкими, но поверхностными социальными контактами.'},
                       {'id': 16, 'category': 'Механизамы защиты', 'short_name': 'конфликт', 'description': 'Тест показывает наличие конфликта, он может иметь различную природу: стресс, тревожность, депрессия, апатия.'},
                       {'id': 17, 'category': 'Механизамы защиты', 'short_name': 'вытеснение', 'description': 'Защитный паттерн - вытеснение - это механизм защиты, при котором неприятные или болезненные воспоминания, мысли или чувства удаляются из сознательного восприятия. Механизм защиты помогает справляться с болезненными или неприятными чувствами и воспоминаниями. Однако может препятствовать эффективному решению проблем и могут привести к нездоровым образам поведения или мышления.'},
                       {'id': 18, 'category': 'Механизамы защиты', 'short_name': 'изоляция', 'description': 'Защитный паттерн - изоляция - это другой механизм защиты, при котором человек отделяет свои чувства от идей или событий, которые вызывают эти чувства. Механизм защиты помогает справляться с болезненными или неприятными чувствами и воспоминаниями. Однако может препятствовать эффективному решению проблем и могут привести к нездоровым образам поведения или мышления.'},
                       {'id': 19, 'category': 'Интеллектуальные возможности', 'short_name': 'высокий', 'description': 'Высокий интеллект. Люди с высоким интеллектом обычно обладают хорошими аналитическими способностями, быстро усваивают новую информацию и способны решать сложные задачи. Они могут видеть связи между различными идеями и концепциями, что позволяет им быть креативными и инновационными. Люди с высоким интеллектом также обычно хорошо справляются с абстрактным мышлением и могут обдумывать сложные идеи без конкретных примеров.'},
                       {'id': 20, 'category': 'Интеллектуальные возможности', 'short_name': 'низкий', 'description': 'Интеллект ниже среднего или интеллектуальная деградация. Люди с низким интеллектом могут испытывать трудности в обучении и понимании новой информации. Они могут столкнуться с проблемами при решении сложных задач и могут нуждаться в дополнительной помощи или поддержке. Люди с низким интеллектом могут также испытывать трудности с абстрактным мышлением и могут предпочитать конкретные примеры и инструкции.'},
                       {'id': 21, 'category': 'Интеллектуальные возможности', 'short_name': 'недостаточно', 'description': 'Интеллектуальные способности используются не в полной мере. Это может произойти по различным причинам, включая недостаток мотивации, отсутствие поддержки или ресурсов, или проблемы с самооценкой. Эти люди могут обладать высоким интеллектом, но не демонстрировать его в полной мере из-за этих препятствий.'},
                   ]
                   )


def downgrade():
    op.drop_table('diagnosis')