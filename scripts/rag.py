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

def qa_ai(qa_chain, text):
    """Обработка запроса к RAG-системе"""
    try:
        result = qa_chain.invoke({"query": text})
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])
        
        # Очистка ответа
        if "Информации недостаточно" in answer:
            answer = "Информации недостаточно."
        elif "\n\n" in answer:
            answer = answer.split("\n\n")[0]
            
        # Обрезка по двойным точкам
        m = re.search(r"\.\s*\.", answer)
        if m:
            answer = answer[:m.start() + 1]
            
        # Форматирование источников
        sources_text = ""
        if sources:
            unique_sources = set()
            for doc in sources:
                meta = getattr(doc, "metadata", {})
                source = meta.get("source", "Неизвестно")
                title = meta.get("title", "")
                unique_sources.add(f"{title} ({source})")
            
            sources_text = "\n".join([f"- {s}" for s in unique_sources])
                
        return answer, sources_text
    except Exception as e:
        print(f"[ERROR] Ошибка в qa_ai: {e}")
        return "Произошла ошибка при обработке запроса", ""

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

# ... остальные функции без изменений
