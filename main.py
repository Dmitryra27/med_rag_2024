# main.py - Medical RAG API
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import vertexai
from vertexai.generative_models import GenerativeModel
import pinecone
import httpx
import traceback
import asyncio

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные ---
gemini_model = None
pinecone_index = None
pinecone_client = None
INITIALIZATION_ERROR = None

# --- Конфигурация ---
PROJECT_ID = os.environ.get("PROJECT_ID", "ai-project-26082025")
REGION = os.environ.get("REGION", "us-central1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-knowledge")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID")
YANDEX_GPT_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest" if YANDEX_FOLDER_ID else None
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", 5000))

# --- Lifespan handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model, pinecone_index, pinecone_client, INITIALIZATION_ERROR
    logger.info("🚀 Начало инициализации приложения...")

    try:
        # Инициализация Vertex AI
        logger.info(f"🔧 Инициализация Vertex AI: project={PROJECT_ID}, location={REGION}")
        vertexai.init(project=PROJECT_ID, location=REGION)

        # Загрузка модели Gemini
        logger.info("🧠 Загрузка модели Gemini...")
        try:
            gemini_model = GenerativeModel("gemini-1.5-pro")
            logger.info("✅ Модель Gemini загружена.")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Gemini: {e}")
            gemini_model = None

        # Инициализация Pinecone
        if PINECONE_API_KEY and PINECONE_INDEX_NAME:
            logger.info("🔗 Инициализация Pinecone...")
            pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
            logger.info(f"✅ Pinecone инициализирован. Индекс: {PINECONE_INDEX_NAME}")
        else:
            logger.warning("⚠️ Pinecone не настроен")

        INITIALIZATION_ERROR = None

    except Exception as e:
        error_msg = f"💥 Ошибка инициализации: {e}"
        logger.error(error_msg)
        logger.error(f"   Traceback: {traceback.format_exc()}")
        INITIALIZATION_ERROR = str(e)

    logger.info("🟢 Приложение готово к обработке запросов.")
    yield
    logger.info("🛑 Завершение работы приложения...")

# --- Создание приложения ---
app = FastAPI(
    lifespan=lifespan,
    title="Medical RAG API",
    version="1.0.0"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных ---
class QuestionRequest(BaseModel):
    question: str
    mode: str = "knowledge_base"

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    mode: str

# --- Вспомогательные функции ---
async def search_knowledge_base(question: str, top_k: int = 3):
    """Поиск в базе знаний Pinecone"""
    global pinecone_index, pinecone_client

    if not pinecone_index or not pinecone_client:
        logger.warning("База знаний недоступна")
        return ["База знаний недоступна"], ["Система"]

    try:
        logger.debug(f"🔍 Поиск: '{question}'")

        # Создание эмбеддинга
        embedding_response = await asyncio.to_thread(
            pinecone_client.inference.embed,
            model="llama-text-embed-v2",
            inputs=[question],
            parameters={"input_type": "query", "truncate": "END"}
        )

        question_embedding = embedding_response.data[0].values

        # Поиск
        search_results = await asyncio.to_thread(
            pinecone_index.query,
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Обработка результатов
        contexts = []
        sources = []

        if search_results.matches:
            for match in search_results.matches:
                metadata = match.metadata or {}
                text = metadata.get('content') or f"Документ ID: {match.id}"
                contexts.append(text)
                source = metadata.get('source', 'Неизвестный источник')
                sources.append(source)
        else:
            contexts.append("Информация не найдена в базе знаний.")
            sources.append("База знаний")

        return contexts, sources

    except Exception as e:
        logger.error(f"❌ Ошибка поиска: {e}")
        return ["Ошибка поиска"], ["Ошибка"]

async def generate_with_gemini(question: str, contexts: List[str]) -> str:
    """Генерация ответа с помощью Google Gemini"""
    if not gemini_model:
        return "Google Gemini недоступен"

    try:
        context_text = "\n\n".join(contexts)
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + "..."

        prompt = f"""
        Ты — медицинский ассистент. Отвечай на основе контекста.
        Контекст: {context_text}
        Вопрос: {question}
        Ответ:
        """.strip()

        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logger.error(f"❌ Ошибка Gemini: {e}")
        return f"Ошибка Gemini: {str(e)}"

async def generate_with_yandex(question: str, contexts: List[str]) -> str:
    """Генерация ответа с помощью Yandex GPT"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return "Yandex GPT не настроен"

    try:
        context_text = "\n\n".join(contexts)
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + "..."

        prompt = f"""
        Ты — медицинский ассистент. Отвечай на основе контекста.
        Контекст: {context_text}
        Вопрос: {question}
        Ответ:
        """.strip()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                'https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
                headers={
                    'Authorization': f'Api-Key {YANDEX_API_KEY}',
                    'x-folder-id': YANDEX_FOLDER_ID,
                    'Content-Type': 'application/json'
                },
                json={
                    'modelUri': YANDEX_GPT_MODEL_URI,
                    'completionOptions': {
                        'stream': False,
                        'temperature': 0.1,
                        'maxTokens': '2000'
                    },
                    'messages': [
                        {'role': 'system', 'text': 'Ты — медицинский ассистент.'},
                        {'role': 'user', 'text': prompt}
                    ]
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data['result']['alternatives'][0]['message']['text'].strip()

    except Exception as e:
        logger.error(f"❌ Ошибка Yandex GPT: {e}")
        return f"Ошибка Yandex GPT: {str(e)}"

def search_open_sources(question: str) -> List[dict]:
    """Поиск в открытых источниках"""
    return [
        {
            "name": "MedlinePlus",
            "content": "Общая медицинская информация от NIH"
        },
        {
            "name": "WHO",
            "content": "Глобальные медицинские рекомендации"
        },
        {
            "name": "CDC",
            "content": "Информация о профилактике заболеваний"
        }
    ]

# --- Эндпоинты ---
@app.get("/")
async def home():
    return {
        "status": "ok",
        "message": "Medical RAG API",
        "models_initialized": {
            "gemini_model": gemini_model is not None,
            "pinecone_index": pinecone_index is not None
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if gemini_model and pinecone_index else "degraded",
        "components": {
            "gemini_model": "healthy" if gemini_model else "uninitialized",
            "pinecone_index": "healthy" if pinecone_index else "uninitialized"
        }
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    mode = request.mode
    logger.info(f"📥 Вопрос: {question} (режим: {mode})")

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    try:
        answer = ""
        sources = []

        if mode == "knowledge_base":
            contexts, sources = await search_knowledge_base(question)
            answer = "Найденная информация:\n\n" + "\n\n---\n\n".join(contexts)

        elif mode == "combined_ai":
            contexts, sources = await search_knowledge_base(question)
            if contexts and contexts != ["База знаний недоступна"]:
                tasks = []
                if gemini_model:
                    tasks.append(generate_with_gemini(question, contexts))
                if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                    tasks.append(generate_with_yandex(question, contexts))

                if tasks:
                    answers = await asyncio.gather(*tasks, return_exceptions=True)
                    answer_parts = []
                    for i, ans in enumerate(answers):
                        model_name = "Google Gemini" if i == 0 and gemini_model else "Yandex GPT"
                        if isinstance(ans, Exception):
                            answer_parts.append(f"❌ Ошибка {model_name}: {str(ans)}")
                        else:
                            answer_parts.append(f"✅ {model_name}:\n{ans}")
                    answer = "\n\n---\n\n".join(answer_parts)
                else:
                    answer = "Ни одна модель ИИ не доступна"
            else:
                answer = "База знаний недоступна"
                sources = ["Система"]

        elif mode == "unified_ai":
            tasks = []
            if gemini_model:
                tasks.append(generate_with_gemini(question, []))
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_with_yandex(question, []))

            if tasks:
                answers = await asyncio.gather(*tasks, return_exceptions=True)
                answer_parts = []
                for i, ans in enumerate(answers):
                    model_name = "Google Gemini" if i == 0 and gemini_model else "Yandex GPT"
                    if isinstance(ans, Exception):
                        answer_parts.append(f"❌ Ошибка {model_name}: {str(ans)}")
                    else:
                        answer_parts.append(f"✅ {model_name}:\n{ans}")
                answer = "\n\n---\n\n".join(answer_parts)
                sources = ["Google Gemini", "Yandex GPT"]
            else:
                answer = "Ни одна модель ИИ не доступна"
                sources = ["Система"]

        elif mode == "open_sources":
            open_sources = search_open_sources(question)
            sources = [src["name"] for src in open_sources]

            open_contexts = [f"Источник: {src['name']}\n{src['content']}" for src in open_sources]
            context_text = "\n\n".join(open_contexts)

            tasks = []
            if gemini_model:
                tasks.append(generate_with_gemini(question, [context_text]))
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_with_yandex(question, [context_text]))

            if tasks:
                answers = await asyncio.gather(*tasks, return_exceptions=True)
                answer_parts = []
                for i, ans in enumerate(answers):
                    model_name = "Google Gemini" if i == 0 and gemini_model else "Yandex GPT"
                    if isinstance(ans, Exception):
                        answer_parts.append(f"❌ Ошибка {model_name}: {str(ans)}")
                    else:
                        answer_parts.append(f"✅ {model_name}:\n{ans}")
                answer = "\n\n---\n\n".join(answer_parts)
            else:
                answer = f"📚 Информация: {context_text}"

        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый режим")

        logger.info("✅ Ответ отправлен")
        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
            mode=mode
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Запуск сервера на порту {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
