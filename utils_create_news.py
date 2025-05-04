from bs4 import BeautifulSoup
import pickle
import spacy
import io
import re

nlp = spacy.load("en_core_web_lg")

def is_nonsense(sent):
    # Удалить лишние пробелы и невидимые символы
    text = sent.text.strip()

    # Удалить слишком короткие или пустые предложения
    if len(text) < 10:
        return True

    # Удалить артефакты вроде многоточий, мусора из повторов
    if re.fullmatch(r'[\.\,\!\?\-\s]{2,}', text):
        return True

    # Удалить предложения без глагола (обычно это неполный или сгенерированный мусор)
    if not any(token.pos_ == "VERB" for token in sent):
        return True

    # Удалить предложения без подлежащего
    if not any(token.dep_ in ("nsubj", "nsubjpass", "expl") for token in sent):
        return True

    return False

def filter_nonsense_sentences(text):
    doc = nlp(text)
    clean_sentences = [sent.text.strip() for sent in doc.sents if not is_nonsense(sent)]
    return ' '.join(clean_sentences)

def phi3_5_text_processing(paper):

    # Удаляем артефакты, связанные с подсчетом кол-ва сгенерированных токенов
    pattern = r'\(Word Count:\s*≈?\s*\d+\s*words?\)|\(Word Count:\s*≈?\s*\d+\s*\)'
    paper = re.sub(pattern, '', paper)

    # Удаляем артефакты, связанные с предупреждением модели о недостоверности текста
    paper = re.sub(r'```.*?```', '', ''.join(paper), flags=re.DOTALL)

    # Удаляем повторную генерацию новости в рамках одной итерации
    paper = paper.split("I'm sorry, I misunderstood your")[0]

    # Удаляем лишние пробелы, артефакты модели, псевдо-предложения
    paper = filter_nonsense_sentences(paper)

    return paper

def remove_tags_bs(html):
    soup = BeautifulSoup(html, "html.parser")

    for aside in soup.find_all('aside'):
        aside.decompose()

    for strong in soup.find_all('strong'):
        strong.decompose()

    return soup.get_text()

def clean_article_text(text: str) -> str:
    """
    Очищает текст новостной статьи от различных технических и рекламных элементов.
    Выполняет следующие преобразования:
    
    1. Удаляет временные метки обновлений (например "Updated at 6.17pm BST")
    2. Удаляет все URL-ссылки (http/https и pic.twitter.com)
    3. Удаляет рекламные вставки и разделители
    4. Удаляет подписи об исправлениях статьи
    5. Удаляет призывы к действию ("contact us", "share here")
    6. Нормализует пробелы и переносы строк
    
    Parameters:
        text (str): Исходный текст статьи
        
    Returns:
        str: Очищенный текст статьи
    """
    # Удаление временных меток обновлений
    text = re.sub(r'Updated at \d{1,2}\.\d{2}(?:am|pm) [A-Z]{3}\b', '', text)
    
    # Удаление URL-ссылок
    text = re.sub(
        r'(?:https?://|pic\.twitter\.com/)\S+|'
        r'\b(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?\b',
        '', 
        text,
        flags=re.IGNORECASE
    )
    
    # Удаление рекламных вставок и разделителей
    text = re.split(r'Guardian Newsroom', text)[0]
    text = text.replace('| ', '').replace('|', '')
    
    # Удаление Related-блоков
    text = re.sub(r'Related:.*?(?=\n|$)', '', text, flags=re.MULTILINE)
    
    # Удаление рекламных блоков
    text = re.sub(r'What\'s on the grid.*', '', text, flags=re.DOTALL)
    
    # Удаление информации об исправлениях
    text = re.sub(
        r'^\s*•?\s*This article was (amended|updated|corrected) on.*?$',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )    
    
    # Удаление призывов к действию
    action_phrases = [
        r'\b(?:see|contribute|share)\b.*?\bhere\b\.?',
        r'This Community callout closed on .*?\.',
        r'Please\s+(?:email|contact|call)\s+us.*?[.!?](?:\s|$)',
        r'Have an opinion.*?[.!?](?:\s|$)'
    ]
    for pattern in action_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Нормализация пробелов и переносов строк
    text = re.sub(r'\n\s*\n', '\n', text)  # Удаление пустых строк
    text = re.sub(r'[ \t]+\n', '\n', text)  # Удаление пробелов перед переносами
    text = re.sub(r'\n[ \t]+', '\n', text)  # Удаление пробелов после переносов
    text = re.sub(r'\n\s*\n', '\n', text)  # Удаление пустых строк
    text = text.strip()
    
    return text

def try_load(file_path:str, file_name:str, client_s3, BUCKET_NAME='graduate'):
    try:
        with open(file_path+file_name, "rb") as file:
            return pickle.load(file)
    except:
        print('Load pdf from s3: original news')

        client_s3.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=file_name,
            file_path=file_path+file_name
            )
        
        with open(file_path+file_name, "rb") as file:
            return pickle.load(file)


def save_s3(pickle_data, object_key, client_s3, BUCKET_NAME='graduate'):

    pickle_data = pickle.dumps(pickle_data)

    client_s3.put_object(
        bucket_name=BUCKET_NAME, 
        object_name=object_key, 
        data=io.BytesIO(pickle_data), 
        length=len(pickle_data), 
        content_type="application/octet-stream"
        )
    
def clean_titile_space(text):

    cleaned_text = re.sub(r'^\s*\*\*.*\*\*\s*$', '', text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*###.*$', '', cleaned_text, flags=re.MULTILINE)

    # Нормализация пробелов и переносов строк
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Удаление пустых строк
    cleaned_text = re.sub(r'[ \t]+\n', '\n', cleaned_text)  # Удаление пробелов перед переносами
    cleaned_text = re.sub(r'\n[ \t]+', '\n', cleaned_text)  # Удаление пробелов после переносов
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)  # Удаление пустых строк
    cleaned_text = cleaned_text.strip()

    return cleaned_text