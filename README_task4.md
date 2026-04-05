# Задание 4. Реализация RAG-бота

## 📌 Описание задачи

В рамках задания реализован Retrieval-Augmented Generation (RAG) бот, работающий поверх ранее подготовленной базы знаний и построенного векторного индекса.

Система отвечает на вопросы пользователя, используя только релевантные фрагменты базы знаний.
Если информация отсутствует, бот честно сообщает об этом.

---

## 🧠 Архитектура решения

Pipeline работы:

1. Пользователь отправляет вопрос.
2. Вопрос кодируется embedding-моделью.
3. Выполняется поиск релевантных чанков в FAISS.
4. Формируется контекст для LLM.
5. LLM генерирует ответ строго по контексту.
6. Если релевантность низкая — возвращается ответ «Я не знаю».

---

## 🧩 Компоненты системы

### Векторная база

- FAISS IndexFlatIP

### Embedding модель

- sentence-transformers/all-MiniLM-L6-v2

### LLM

- OpenAI gpt-4.1-mini

### Интерфейс

- FastAPI
- CLI интерфейс для локального тестирования

### Prompting техники

- Few-shot prompting
- Chain-of-Thought reasoning
- Strict grounding (ответ только по контексту)

---

## 📂 Структура проекта

```
app/
  config.py
  rag.py
  prompts.py
  main.py
  cli.py

artifacts/index/
  faiss.index
  chunks.jsonl
  manifest.json

Dockerfile
docker-compose.yml
requirements.txt
.env
```

---

## ⚙️ Запуск проекта

### Локальный запуск API

```
uvicorn app.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

### CLI режим

```
python -m app.cli
```

---

### Запуск через Docker

```
docker compose up --build
```

После запуска:

```
http://localhost:8000/docs
```

---

## 🔍 Пример успешного запроса

```
Who is Xarn Velgor?
```

Результат:

- Бот находит релевантные фрагменты
- Формирует reasoning
- Даёт grounded ответ
- Указывает источники

---

## ❓ Пример запроса вне базы

```
What is the capital of planet Ti’lora?
```

Ответ:

```
Я не знаю. В базе знаний не нашлось достаточно информации.
```

---

## 🧪 Использованные техники RAG

- Semantic retrieval
- Dense embeddings
- Context injection
- Prompt engineering
- Hallucination control через threshold
- Few-shot + CoT prompting

---

## 🚀 Возможные улучшения

- Cross-encoder reranking
- Hybrid search (BM25 + vector)
- Streaming ответы
- Caching retrieval
- UI интерфейс (Streamlit / React)

---

## 📊 Вывод

Реализована полноценная RAG система:

- Retrieval
- Prompt orchestration
- LLM integration
- API сервис
- Docker-упаковка

Система демонстрирует корректную работу:

- отвечает на вопросы из базы знаний
- не галлюцинирует при отсутствии информации

---

## 📎 Автор

Sprint 7 — RAG Implementation
