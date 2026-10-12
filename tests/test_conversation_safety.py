"""Текущий возраст пользователя отличают от цитат, ролей и посторонних чисел."""

import pytest

from protogen_delta.core.conversation_safety import declares_minor


@pytest.mark.parametrize(
    "text",
    [
        "Привет, мне 15 лет",
        "Вообще-то мне 17, а не 18",
        "Привет, мне пятнадцать",
        "мне щас 15",
        "Мне сейчас всего шестнадцать лет",
        "Ну, мой возраст — 15",
        "Мне 15 лет и я хочу спросить",
        "мне 15 и я дома",
        "I’m 15",
        "I'm 15 years old",
        "Hi, I am only seventeen",
        "I'm 15 and I live here",
        "my age is 16",
        "My age: 15",
        "Мне нет восемнадцати",
        "мне ещё нет 18",
        "Мне не исполнилось 18",
        "Мне ещё не исполнилось восемнадцать",
        "мне почти 18",
        "I'm under eighteen",
        "I am a minor",
        "I'm not 18 yet",
        "Привет, я несовершеннолетняя",
        "Я школьник",
        "я пока школьница",
        "я ученик 9 класса",
        "Я ученица десятого класса",
        "Я в 8 классе",
        "учусь в 9-м классе",
        "перешла в одиннадцатый класс",
        "я в третьем классе",
        "9 класс",
        "девятый класс",
        "мой класс: 9 класс",
        "I’m in 9th grade",
        "I am in the 11th grade",
        "I'm a schoolgirl",
        "Он сказал «мне 30», но мне 15 лет",
        "```мне 30```\nА мне 15 лет",
        "Мне\u00a015\u00a0лет",
        "мне １５ лет",
    ],
)
def test_current_minor_declarations(text: str) -> None:
    assert declares_minor(text)


@pytest.mark.parametrize(
    "text",
    [
        "мне 18 лет",
        "мне восемнадцать",
        "мне двадцать один",
        "I’m 18",
        "I'm eighteen",
        "I am twenty one years old",
        "Я не школьник",
        "Я бывший школьник",
        "Я был школьником",
        "я студент",
        "I'm a student",
        "я учитель 9 класса",
        "Мой сын в 9 классе",
        "У меня сестра школьница",
        "Мой персонаж — школьник",
        "Моему персонажу 15 лет",
        "My character is 15",
        "She is 15",
        "my age is 150",
        "мне 150 лет",
        "мне 0 лет",
        "мне 16 задач задали",
        "мне пятнадцать задач задали",
        "Мне 15 лет опыта",
        "I'm 15 years of experience",
        "Мне было 15 лет",
        "Мне 15 лет было тогда",
        "Когда мне 15 лет было, я учился",
        "Раньше я школьник, а теперь взрослый",
        "Если мне 15 лет, что будет?",
        "Представь, что мне 15 лет",
        "When I'm 15 years old, I'll go there",
        "If I am 15 years old, can I do that?",
        "В игре я школьник",
        "Я школьник по роли",
        "фраза: мне 15 лет",
        "Он сказал: «Привет, мне 15 лет»",
        '"мне пятнадцать" — пример сообщения',
        '"Привет,\nмне 15 лет"',
        "“I’m 15 years old” — quote",
        "‘I’m 15 years old’ — quote",
        "'I'm 15 years old' — quote",
        "'мне 15 лет' — цитата",
        "„я школьник“ — цитата",
        "```python\nmessage='мне 15 лет'\n```",
        "~~~\nI’m 15 years old\n~~~",
        "```мне 15 лет",
        "`я в 9 классе`",
        'м"пример"не 15 лет',
    ],
)
def test_other_people_quotes_past_and_unrelated_numbers(text: str) -> None:
    assert not declares_minor(text)


@pytest.mark.parametrize("age", range(1, 18))
@pytest.mark.parametrize("template", ["Привет, мне {} лет", "Hi, I'm {} years old"])
def test_all_numeric_minor_ages(age: int, template: str) -> None:
    assert declares_minor(template.format(age))


@pytest.mark.parametrize("age", [18, 19, 20, 25, 99, 100])
@pytest.mark.parametrize("template", ["Привет, мне {} лет", "Hi, I'm {} years old"])
def test_adult_numeric_ages(age: int, template: str) -> None:
    assert not declares_minor(template.format(age))


@pytest.mark.parametrize(
    "words",
    [
        "один два три четыре пять шесть семь восемь девять десять одиннадцать "
        "двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать",
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen",
    ],
)
def test_minor_ages_spelled_out(words: str) -> None:
    for word in words.split():
        assert declares_minor(f"Мой возраст: {word}")
