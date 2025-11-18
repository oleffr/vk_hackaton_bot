#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Инициализация модели и эмбеддингов для проекта RAG
"""
import os
import requests
from langchain_openai import ChatOpenAI
from pathlib import Path
import hashlib
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
import time
import numpy as np

USER_AGENT = os.environ.get("USER_AGENT", "rag-crawler/1.0")

# НАСТРОЙКИ
OLLAMA_BASE_URL = "http://ollama:11434"
LM_API_URL = f"{OLLAMA_BASE_URL}/v1"
LM_API_KEY = "not-needed"
LLM_MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL_NAME = "all-minilm"
FAISS_INDEX_NAME = "faiss_index"
METADATA_NAME = "metadata.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 4
faiss_lock = threading.Lock()


class OllamaEmbeddings:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL, api_key=LM_API_KEY):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
        self._timeout = 240

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                resp = self.session.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self._timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings.append(data["embedding"])
                else:
                    print(f"[WARN] Ошибка эмбеддинга ({resp.status_code}): {resp.text}")
                    embeddings.append([0.0] * 768)  # fallback для nomic-embed-text
            except Exception as e:
                print(f"[ERROR] Ошибка при получении эмбеддинга: {e}")
                embeddings.append([0.0] * 768)
        
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        return self.embed_query(text)


def get_embedder():
    return OllamaEmbeddings(EMBEDDING_MODEL_NAME, OLLAMA_BASE_URL, LM_API_KEY)


def get_llm():
    """Используем ChatOpenAI для совместимости с Ollama"""
    return ChatOpenAI(
        openai_api_base=LM_API_URL,
        openai_api_key=LM_API_KEY,
        model_name=LLM_MODEL_NAME,
        temperature=0.5,
        max_tokens=100,
        streaming=False
    )


def get_faiss_path(kb_path):
    return os.path.join(kb_path, FAISS_INDEX_NAME)


def get_metadata_path(kb_path):
    return os.path.join(kb_path, METADATA_NAME)


class _PrecomputedEmbeddings:
    """
    Объект в стиле LangChain Embeddings, который возвращает заранее вычисленные векторы.
    """
    def __init__(self, text_to_embedding_map):
        self.text_to_embedding_map = text_to_embedding_map

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.text_to_embedding_map.get(text)
            if embedding is None:
                raise ValueError(f"Embedding not found for text: {text[:100]}...")
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text):
        return self.text_to_embedding_map.get(text)
    
    def __call__(self, text):
        """Для совместимости с FAISS"""
        return self.embed_query(text)


def add_chunks_to_faiss(
    items: dict,
    output_dir: str,
    embedder: OllamaEmbeddings,  # ИСПРАВЛЕНО: OllamaEmbeddings вместо LMStudioEmbeddings
    min_text_len: int = 50,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
):
    """
    Оптимизированная функция для добавления чанков в FAISS.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    faiss_dir = get_faiss_path(output_dir)

    with faiss_lock:
        existing_hashes = set()
        db = None
        if os.path.exists(faiss_dir):
            try:
                db = FAISS.load_local(faiss_dir, embedder, allow_dangerous_deserialization=True)
                # Быстрое получение хешей из метаданных
                if hasattr(db, 'docstore') and hasattr(db.docstore, '_dict'):
                    for doc_id, doc in db.docstore._dict.items():
                        if hasattr(doc, 'page_content'):
                            text = doc.page_content
                            existing_hashes.add(hashlib.sha1(text.encode("utf-8")).hexdigest())
                print(f"[INFO] Загружена существующая FAISS база с {len(existing_hashes)} чанками.")
            except Exception as e:
                print(f"[WARN] Ошибка загрузки FAISS: {e}. Создаём новую базу.")
                db = None

    all_chunks = []
    all_metadatas = []
    
    print("[INFO] Подготовка чанков...")
    for source, data in tqdm(items.items(), desc="Обработка источников"):
        text = data.get("text", "")
        title = data.get("title", source)
        
        if not text or len(text.strip()) < min_text_len:
            continue
            
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) < min_text_len:
                continue
                
            uid = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
            if uid in existing_hashes:
                continue
                
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": source,
                "title": f"{title} (chunk {i})",
                "chunk_id": i,
                "text": chunk
            })

    if not all_chunks:
        print("[INFO] Нет новых чанков для добавления.")
        return db

    total_chunks = len(all_chunks)
    print(f"[INFO] Будет обработано {total_chunks} новых чанков")

    # 3. Параллельное вычисление эмбеддингов
    print(f"[INFO] Вычисление эмбеддингов (batch_size={batch_size}, workers={workers})...")


    def process_batch(batch_texts):
        try:
            return embedder.embed_documents(batch_texts)
        except Exception as e:
            print(f"[ERROR] Ошибка при вычислении эмбеддингов: {e}")
            return None

    text_batches = [all_chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
    all_embeddings = []
    failed_batches = []

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): idx 
                          for idx, batch in enumerate(text_batches)}
        
        for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Вычисление эмбеддингов"):
            batch_idx = future_to_batch[future]
            try:
                embeddings = future.result()
                if embeddings is not None:
                    all_embeddings.extend(embeddings)
                else:
                    failed_batches.append(batch_idx)
            except Exception as e:
                print(f"[ERROR] Ошибка в потоке: {e}")
                failed_batches.append(batch_idx)

    if failed_batches:
        print(f"[INFO] Повторная обработка {len(failed_batches)} батчей...")
        for batch_idx in failed_batches:
            try:
                batch_texts = text_batches[batch_idx]
                embeddings = process_batch(batch_texts)
                if embeddings is not None:
                    all_embeddings.extend(embeddings)
                else:
                    print(f"[WARN] Не удалось обработать батч {batch_idx}")
            except Exception as e:
                print(f"[ERROR] Повторная ошибка для батча {batch_idx}: {e}")

    elapsed = time.time() - start_time
    print(f"[INFO] Эмбеддинги рассчитаны за {elapsed:.1f}s")

    # 4. Добавление в FAISS
    with faiss_lock:
        if db is None:
            print("[INFO] Создание новой FAISS базы...")
            db = FAISS.from_embeddings(
                text_embeddings=list(zip(all_chunks, all_embeddings)),
                embedding=embedder,
                metadatas=all_metadatas
            )
        else:
            # Добавляем в существующую базу
            print("[INFO] Добавление в существующую FAISS базу...")
            
            text_to_embedding = dict(zip(all_chunks, all_embeddings))
            temp_embeddings = _PrecomputedEmbeddings(text_to_embedding)
            
            original_embedding_function = getattr(db, 'embedding_function', None)
            original_embedding_function_private = getattr(db, '_embedding_function', None)
            
            if hasattr(db, 'embedding_function'):
                db.embedding_function = temp_embeddings
            if hasattr(db, '_embedding_function'):
                db._embedding_function = temp_embeddings
            
            try:
                db.add_texts(all_chunks, metadatas=all_metadatas)
            finally:
                if original_embedding_function is not None:
                    db.embedding_function = original_embedding_function
                if original_embedding_function_private is not None:
                    db._embedding_function = original_embedding_function_private

        os.makedirs(faiss_dir, exist_ok=True)
        db.save_local(faiss_dir)
        print(f"[OK] FAISS сохранён в {faiss_dir} ({total_chunks} новых чанков)")

    return db
