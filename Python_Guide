## Локальный запуск
**ВНИМАНИЕ! Для локального запуска требуется модель скачанная модель ollama**
### Общий синтаксис локального запуска

```bash
python main.py <command> [options]
```

### Доступные команды

1. **update_pdf** – добавить или обновить PDF в базе знаний
   ```bash
   python main.py pdf --pdf_dir ./pdfs --out ./kb_output
   ```
   **Ньюансы:** 
   - Добавляет только новые PDF-файлы или новые чанки текста.
   - Сохраняет метаданные для каждого документа.
   - Использует LM Studio для генерации эмбеддингов.

2. **update_url** – добавить или обновить веб-страницы в базе знаний
   ```bash
   python main.py url --seeds ./seed_urls.txt --out ./kb_output --max_pages 200 --delay 0.1
   ```
   **Ньюансы:** 
   - Рекурсивно обходит страницы, ограничено `--max-pages`.
   - Игнорирует внешние домены, email, телефоны.
   - Обновляет существующую FAISS базу без дублирования чанков.
   - Создаёт/дополняет `urls.txt` с уникальными URL.

3. **chat** – запустить RAG чат-бота
   ```bash
   python main.py chat --out ./kb_output
   ```
   **Ньюансы:** 
   - Использует текущую FAISS базу.
   - Возвращает ответы с источниками (URL или PDF) по чанкам.
   - Поддерживает exit/выход/quit для завершения.

4. **help** – вывод справки
   ```bash
   python main.py help
   ```
   
### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Пример полного рабочего процесса

```bash
# Обновляем базу из PDF
python main.py pdf --pdf_dir ./pdfs --out ./kb_outpu

# Обновляем базу из URL
python main.py url --seeds ./seed_urls.txt --out ./kb_output --max_pages 200 --delay 0.1

# Запускаем чат-бот
python main.py chat --out ./kb_output
```
