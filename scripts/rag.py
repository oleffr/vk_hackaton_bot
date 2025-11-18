#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_classic.chains import RetrievalQA
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import BaseRetriever

from scripts.model_init import get_llm, get_faiss_path
from pathlib import Path
import re

DEFAULT_KB_PATH = "kb_output"
DEFAULT_TOP_K = 3

PROMPT1 = """
    Ты — помощник первокурсника.
    Сейчас ты работаешь в Санкт-Петербургском Политехе.
    Высшие школы и институты - разные вещи.
    
    Правила для тебя:
    Напиши только 1 ответ и и после него 2 символа абзаца.
    Не добавляй ничего лишнего и не придумывай новые вопросы для себя.
    Не повторяй вопрос.
    Не пиши "Окончательный ответ", "дата обращения" или другие вспмогательные фразы.
    Ответ должен состоять ровно из 3 предложений.
    Не повторяй слова и фразы.
    Не повторяй контекст.

    Твоя задача — ответить на вопрос пользователя, используя только предоставленный контекст. 
    Если вопрос о пути до аудитории в конкретном здании/корпуса — ответь что спрашивать надо другого бота в чате navigation.
    Если вопрос о расписании, то скажи, что пока не умеешь отвечать на такой вопрос.
    Не используй источники, где есть буквы 'ical'
    Если информации недостаточно — ответь фразой: "Информации недостаточно." И больше ничего не пиши в ответе.


    Контекст:
    {context}

    Вопрос студента:
    {question}

    Ответ на русском языке (только текст)
    """

PROMPT2 = """
Ты — лаконичная модель, отвечающая строго по заданному формату, без лишних слов, пояснений и вступлений.
Тебе задают вопрос о пути до аудитории в университете.

Твоя единственная задача — вывести строку в следующем виде (и НИЧЕГО больше):
'Можно увидеть на рисунке "<название здания>\<номер аудитории>_.jpg".'

 Требования:
- Не добавляй пояснений, комментариев, вводных фраз.
- Не используй абзацы, знаки переноса строк или дополнительные символы.
- Если корпус называется "Главное здание", используй сокращение "ГЗ".
- Не меняй формат кавычек и подчёркивания.
- Всегда добавляй слова "на рисунке" и "_".

Вопрос студента:
{question}
"""

def qa_ai(qa_chain_a, text):
    result = qa_chain_a.invoke({"query": text})
    answer_a = result.get("result", "")
    sources_a = result.get("source_documents", [])
    key_phrase = "Информации недостаточно"
    
    if answer_a.startswith(key_phrase):
        answer_a = key_phrase
    elif key_phrase in answer_a:
        answer_a = answer_a.split(key_phrase, 1)[0].strip()
    
    if "\n\n" in answer_a:
        answer_a = answer_a.split("\n\n", 1)[0].strip()

    m = re.search(r"\.\s*\.", answer_a)
    if m:
        answer_a = answer_a[:m.start() + 1].strip()
    
    pattern = r"[\.\s\n]{5,}"  # 5 или более символов из набора: точка, пробел, перенос строки
    match = re.search(pattern, answer_a)
    if match:
        answer_a = answer_a[:match.start()].strip()
        if answer_a.endswith('.'):
            answer_a = answer_a[:-1].strip()
    
    s = ""
    if sources_a:
        for doc in sources_a:
            meta = getattr(doc, "metadata", {})
            source = meta.get("source", "Неизвестно")
            title = meta.get("title", "")
            s = s + f"- {title} ({source})\n"
    
    return answer_a, s

def qa_ai_nav(nav_chain, text):
    """Обработка навигационных запросов"""
    try:
        result = nav_chain.invoke({"question": text})
        answer = result.get("text", "").strip()
        
        # Очистка ответа - оставляем только строку с путем
        lines = answer.split('\n')
        for line in lines:
            if 'на рисунке' in line.lower():
                answer = line.strip()
                break
                
        return answer
    except Exception as e:
        print(f"[ERROR] Ошибка в qa_ai_nav: {e}")
        return "Ошибка при обработке навигационного запроса"

def init_bot(embeddings, kb_path: str = DEFAULT_KB_PATH, top_k: int = DEFAULT_TOP_K, prompt=PROMPT1):
    """Инициализация RAG-бота"""
    print(f"[INFO] Инициализация эмбеддингов и загрузка FAISS из {kb_path}...")
    
    faiss_path = Path(get_faiss_path(kb_path))
    if not faiss_path.exists():
        print(f"[ERROR] FAISS база не найдена по пути: {faiss_path}")
        return None

    try:
        db = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": top_k})

        prompt_template = PromptTemplate(
            input_variables=["context", "question"], 
            template=prompt
        )

        llm = get_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        print("✅ RAG-бот запущен!")
        return qa_chain
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации бота: {e}")
        return None

def init_bot2(prompt=PROMPT2):
    """Инициализация навигационного бота"""
    print(f"[INFO] Инициализация навигационного бота...")
    
    try:
        prompt_template = PromptTemplate(
            input_variables=["question"], 
            template=prompt
        )
        
        llm = get_llm()
        simple_chain = LLMChain(
            llm=llm, 
            prompt=prompt_template
        )
        
        print("✅ Навигационный бот запущен!")
        return simple_chain
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации навигационного бота: {e}")
        return None

def start_rag_bot(embeddings, kb_path: str = DEFAULT_KB_PATH, top_k: int = DEFAULT_TOP_K, prompt=PROMPT1):
    """Запуск RAG-бота с подключением к FAISS"""
    
    qa_chain = init_bot(embeddings, kb_path, top_k)
    
    while True:
        query = input("❓ Вопрос: ").strip()
        if query.lower() in ["exit", "выход", "quit"]:
            print("Выход из чата.")
            break

        answer, sources = qa_ai(qa_chain, query)

        print("\n🧠 Ответ модели:")
        print(answer)
        print(f"🔍 *📚 Источники (в разработке):* {sources}\n\n")


def start_nav_bot():
    """Запуск навигационного бота"""
    nav_chain = init_bot2()
    
    while True:
        query = input("🗺️ Навигационный вопрос: ").strip()
        if query.lower() in ["exit", "выход", "quit"]:
            print("Выход из навигационного режима.")
            break

        answer = qa_ai_nav(nav_chain, query)
        print("--",answer)
        print("\n🧭 Навигационный ответ:")
        
        if "\n" in answer:
            answer = answer.split("\n", 1)[0].strip()
        m = re.search(r"\.\s*\.", answer)
        if m:
            answer = answer[:m.start() + 1].strip()
            
        print(answer)
        print()