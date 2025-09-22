# main.py - Medical RAG API с использованием встроенной модели Pinecone llama-text-embed-v2
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import vertexai
from vertexai.generative_models import GenerativeModel
import pinecone
import httpx
import json
import traceback
import asyncio
from datetime import datetime, timezone

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO, # Уровень INFO для production, DEBUG для разработки
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные для хранения инициализированных клиентов ---
gemini_model = None
pinecone_index = None
pinecone_client = None # Новый клиент Pinecone для создания эмбеддингов
INITIALIZATION_ERROR = None # Переменная для хранения ошибки инициализации

# --- Конфигурация ---
PROJECT_ID = os.environ.get("PROJECT_ID", "ai-project-26082025")
REGION = os.environ.get("REGION", "us-central1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-knowledge")
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID",'b1gatnfegvh5a9a5iovu')
YANDEX_GPT_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest" if YANDEX_FOLDER_ID else None
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", 5000)) # Ограничение длины контекст
google_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
# --- Lifespan handler для FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model, pinecone_index, pinecone_client, INITIALIZATION_ERROR
    logger.info("🚀 Начало инициализации приложения Medical RAG API...")

    try:
        # 1. Инициализация Vertex AI (для генерации ответов)
        logger.info(f"🔧 Инициализация Vertex AI: project={PROJECT_ID}, location={REGION}")
        # Проверяем наличие credentials в переменной окружения как JSON string
        google_credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

        if google_credentials_json:
            # Используем credentials из JSON string
            logger.info("🔐 Использование credentials из переменной окружения GOOGLE_APPLICATION_CREDENTIALS_JSON")
            try:
                import json
                from google.oauth2 import service_account

                # Парсим JSON credentials
                credentials_info = json.loads(google_credentials_json)
                credentials = service_account.Credentials.from_service_account_info(credentials_info)

                vertexai.init(
                    project=PROJECT_ID,
                    location=REGION,
                    credentials=credentials
                )
                logger.info("✅ Vertex AI инициализирован с service account credentials")
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации с service account credentials: {e}")
                # fallback на default credentials
                logger.info("🔄 Использование default credentials как резервный вариант")
                vertexai.init(project=PROJECT_ID, location=REGION)
        else:
            # Используем default credentials (если запущено в Google Cloud)
            logger.info("🔐 Использование default credentials")
            vertexai.init(project=PROJECT_ID, location=REGION)
        # 2. Инициализация модели Google Gemini (для генерации ответов)
        logger.info("🧠 Загрузка модели генерации gemini-2.5-pro...")
        try:

            gemini_model = GenerativeModel("gemini-2.5-pro")
        except Exception as e25:
            logger.info("⚠️  Модель gemini-2.5-pro недоступна, пробуем gemini-2.5-pro...")
            try:
                gemini_model = GenerativeModel("gemini-2.5-pro")
            except Exception as e15:
                logger.error(f"❌ Ошибка загрузки модели Gemini 2.5-pro: {e15}")
                gemini_model = None
        if gemini_model:
            logger.info("✅ Модель генерации Gemini загружена.")
        else:
            logger.error("❌ Не удалось загрузить ни одну модель Gemini.")

        # 3. Инициализация Pinecone (для поиска и создания эмбеддингов)
        logger.info("🔗 Инициализация Pinecone...")
        if not PINECONE_API_KEY:
            raise ValueError("❌ Переменная окружения PINECONE_API_KEY не установлена!")
        if not PINECONE_INDEX_NAME:
            raise ValueError("❌ Переменная окружения PINECONE_INDEX_NAME не установлена!")

        # Создаем клиент Pinecone
        pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

        # Подключаемся к индексу
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)

        # Проверка подключения и получения статистики
        index_stats = pinecone_index.describe_index_stats()
        logger.info(f"✅ Pinecone инициализирован. Индекс '{PINECONE_INDEX_NAME}' содержит {index_stats.get('total_vector_count', 0)} векторов.")
        logger.info(f"   Размерность векторов: {index_stats.get('dimension', 'N/A')}")
        logger.info(f"   Используемая модель эмбеддингов: llama-text-embed-v2")

        INITIALIZATION_ERROR = None # Сброс ошибки инициализации

    except Exception as e:
        error_msg = f"💥 Критическая ошибка инициализации: {e}"
        logger.error(error_msg)
        logger.error(f"   Traceback: {traceback.format_exc()}")
        # Сохраняем ошибку, чтобы показать её в /health
        INITIALIZATION_ERROR = str(e)
        # Не бросаем исключение, чтобы приложение запустилось, но с ошибкой

    logger.info("🟢 Приложение готово к обработке запросов.")
    yield
    logger.info("🛑 Завершение работы приложения...")

# --- Создание приложения FastAPI с lifespan handler ---
app = FastAPI(
    lifespan=lifespan,
    title="Medical RAG API",
    version="3.4.0-Pinecone-Llama-Only",
    description="API для медицинского ассистента с RAG, использующий встроенную модель Llama от Pinecone"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Для разработки. В production укажите конкретные origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Глобальный обработчик исключений ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик всех не пойманных исключений."""
    logger.error(f"🚨 Необработанное исключение: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера. Пожалуйста, попробуйте позже.",
            "error_type": type(exc).__name__,
            # В production НЕ показывайте traceback пользователю!
            # "debug_info": traceback.format_exc()
        },
    )

# --- Модели данных ---
class QuestionRequest(BaseModel):
    question: str
    mode: str = "knowledge_base" # 'knowledge_base', 'combined_ai', 'unified_ai', 'open_sources', 'complete_analysis'

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    mode: str

# --- Вспомогательные функции ---
async def search_knowledge_base(question: str, top_k: int = 3):
    """Асинхронный поиск по базе знаний Pinecone с использованием встроенной модели Llama."""
    global pinecone_index, pinecone_client

    if not pinecone_index:
        logger.warning("База знаний недоступна (индекс Pinecone не инициализирован)")
        return ["База знаний недоступна"], ["Система"]

    try:
        logger.debug(f"🔍 Поиск в Pinecone по запросу: '{question}'...")

        # --- 1. Создание эмбеддинга для вопроса с помощью Inference API Pinecone ---
        logger.debug("🧠 Создание эмбеддинга для вопроса с помощью Pinecone Inference API...")

        # Оборачиваем блокирующий вызов в asyncio.to_thread
        embedding_response = await asyncio.to_thread(
            pinecone_client.inference.embed,
            model="llama-text-embed-v2",
            inputs=[question], # Список текстов
            parameters={
                "input_type": "query", # Тип входных данных - запрос
                "truncate": "END"      # Как обрезать, если текст слишком длинный
            }
        )

        # Извлекаем значения вектора
        question_embedding = embedding_response.data[0].values
        logger.debug(f"   Эмбеддинг создан (размерность: {len(question_embedding)})")

        # --- 2. Поиск по векторной базе данных (асинхронно) ---
        logger.debug("🔎 Выполнение поиска в векторной базе данных...")

        # Оборачиваем блокирующий вызов query в asyncio.to_thread
        search_results = await asyncio.to_thread(
            pinecone_index.query,
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True
        )
        logger.debug(f"   Поиск завершен. Найдено {len(search_results.matches)} результатов.")

        # --- 3. Обработка результатов ---
        contexts = []
        sources = []

        if search_results.matches:
            for match in search_results.matches:
                metadata = match.metadata or {}
                # Адаптируйте ключи под структуру ваших метаданных в Pinecone
                # Например: 'content', 'text', 'chunk_text', 'preview'
                text = metadata.get('content') or metadata.get('text') or metadata.get('chunk_text') or metadata.get('preview') or f"Документ ID: {match.id}"
                contexts.append(text)

                source = metadata.get('source', 'Неизвестный источник')
                sources.append(source)
        else:
            logger.info("   Ничего не найдено в базе знаний.")
            contexts.append("Извините, в базе знаний не найдено информации по вашему вопросу.")
            sources.append("База знаний")

        logger.debug(f"   Обработано {len(contexts)} релевантных результатов.")
        return contexts, sources

    except Exception as e:
        logger.error(f"❌ Ошибка поиска в Pinecone: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return ["Ошибка поиска в базе знаний"], ["Pinecone"]

async def generate_with_gemini(question: str, contexts: List[str]) -> str:
    """Генерация ответа с помощью Google Gemini."""
    if not gemini_model:
        logger.warning("Модель Google Gemini недоступна")
        return "Модель Google Gemini недоступна"

    try:
        context_text = "\n\n".join(contexts)
        # Ограничиваем длину контекста
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + f"... (контекст усечен до {MAX_CONTEXT_LENGTH} символов)"
            logger.info(f"   Контекст усечен до {MAX_CONTEXT_LENGTH} символов.")

        prompt = f"""
        Ты — медицинский ассистент.
Отвечай только на основе предоставленного контекста.

🎯 Основная задача:
Максимально использовать всю релевантную информацию из контекста.
Не уходить от темы — строго следуй исходному вопросу.
Формулируй ответ естественно, кратко, по-человечески, как компетентный специалист.
Если в контексте нет данных — скажи прямо:
«В моей базе знаний нет информации по этому вопросу, могу поискать актуальную информацию» — и предложи поиск.
Если информации недостаточно — задай уточняющий вопрос или предложи:
«Хочешь, я поищу дополнительную информацию по этому вопросу?»
Если пользователь говорит «да» — выполни расширенный поиск, дай четкий, структурированный ответ (по пунктам, если уместно), без лишних общих фраз.
✅ Правила:
Никаких шаблонных отказов: не пиши "Это медицинский вопрос", "Я не врач" и т.п.
Не вводи новые темы: не обсуждай то, о чём не спрашивали. Перед поиском еще раз посмотри запрос
Если в контексте нет данных — скажи прямо:
«В моей базе знаний нет информации по этому вопросу, могу поискать актуальную информацию» — и предложи поиск.
При противоречиях в контексте — укажи: «Существуют разные мнения» и кратко объясни каждое.
После расширенного поиска:
Ответ должен быть конкретным, полным, глубоким.
Подавай информацию по пунктам, если это улучшает ясность.
Без рекомендаций типа "обращайтесь к мастерам" или "соблюдайте гигиену" — только факты, если они напрямую относятся к вопросу.
В конце каждого расширенного ответа спроси:
«Нужно ли что-то уточнить или найти дополнительно?»
      Контекст:
        {context_text}

        Вопрос: {question}
        Ответ:
        """.strip()

        logger.debug("💬 Генерация ответа с помощью Google Gemini...")
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        logger.debug("✅ Ответ от Google Gemini сгенерирован.")
        return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Google Gemini: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return f"Ошибка генерации Google Gemini: {str(e)}"

async def generate_with_yandex(question: str, contexts: List[str]) -> str:
    """Генерация ответа с помощью Yandex GPT."""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        logger.warning("Yandex GPT не настроен")
        return "Yandex GPT не настроен"

    try:
        context_text = "\n\n".join(contexts)
        # Ограничиваем длину контекста
        if len(context_text) > MAX_CONTEXT_LENGTH:
            context_text = context_text[:MAX_CONTEXT_LENGTH] + f"... (контекст усечен до {MAX_CONTEXT_LENGTH} символов)"
            logger.info(f"   Контекст усечен до {MAX_CONTEXT_LENGTH} символов.")

        prompt = f"""
                Ты — медицинский ассистент.
Отвечай только на основе предоставленного контекста.

🎯 Основная задача:
Максимально использовать всю релевантную информацию из контекста.
Не уходить от темы — строго следуй исходному вопросу.
Формулируй ответ естественно, кратко, по-человечески, как компетентный специалист.
Если в контексте нет данных — скажи прямо:
«В моей базе знаний нет информации по этому вопросу, могу поискать актуальную информацию» — и предложи поиск.
Если информации недостаточно — задай уточняющий вопрос или предложи:
«Хочешь, я поищу дополнительную информацию по этому вопросу?»
Если пользователь говорит «да» — выполни расширенный поиск, дай четкий, структурированный ответ (по пунктам, если уместно), без лишних общих фраз.
✅ Правила:
Никаких шаблонных отказов: не пиши "Это медицинский вопрос", "Я не врач" и т.п.
Не вводи новые темы: не обсуждай то, о чём не спрашивали. Перед поиском еще раз посмотри запрос
Если в контексте нет данных — скажи прямо:
«В моей базе знаний нет информации по этому вопросу, могу поискать актуальную информацию» — и предложи поиск.
При противоречиях в контексте — укажи: «Существуют разные мнения» и кратко объясни каждое.
После расширенного поиска:
Ответ должен быть конкретным, полным, глубоким.
Подавай информацию по пунктам, если это улучшает ясность.
Без рекомендаций типа "обращайтесь к мастерам" или "соблюдайте гигиену" — только факты, если они напрямую относятся к вопросу.
В конце каждого расширенного ответа спроси:
«Нужно ли что-то уточнить или найти дополнительно?»
      Контекст:
        {context_text}

        Вопрос: {question}
        Ответ:
        """.strip()

        logger.debug("💬 Генерация ответа с помощью Yandex GPT...")
        async with httpx.AsyncClient() as client:
            yandex_response = await client.post(
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
            yandex_response.raise_for_status()
            yandex_data = yandex_response.json()
            answer = yandex_data['result']['alternatives'][0]['message']['text'].strip()
            logger.debug("✅ Ответ от Yandex GPT сгенерирован.")
            return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Yandex GPT: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return f"Ошибка генерации Yandex GPT: {str(e)}"

# Новые вспомогательные функции для разных режимов
async def generate_without_context(question: str) -> str:
    """Генерация ответа без контекста (общие медицинские знания)"""
    if not gemini_model:
        return "Модель недоступна"

    prompt = f"""
    Ты — медицинский ассистент. Ответь на следующий вопрос, используя свои общие медицинские знания.
    Отвечай профессионально, но понятно.
    Если не уверен в ответе, честно это скажи.
    
    Вопрос: {question}
    Ответ:
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Ошибка генерации без контекста: {e}")
        return f"Ошибка: {str(e)}"

async def generate_without_context_yandex(question: str) -> str:
    """Генерация ответа без контекста через Yandex GPT"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return "Yandex GPT не настроен"

    prompt = f"""
    Ты — медицинский ассистент. Ответь на следующий вопрос, используя свои общие медицинские знания.
    Отвечай профессионально, но понятно.
    Если не уверен в ответе, честно это скажи.
    
    Вопрос: {question}
    Ответ:
    """

    try:
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
                        'temperature': 0.7,
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
        logger.error(f"Ошибка генерации без контекста Yandex: {e}")
        return f"Ошибка: {str(e)}"

async def generate_with_context(question: str, context: str) -> str:
    """Генерация ответа с контекстом через Gemini"""
    if not gemini_model:
        return "Модель недоступна"

    prompt = f"""
    Ты — медицинский ассистент. Проанализируй информацию из источников и ответь на вопрос.
    
    Информация из источников:
    {context}
    
    Вопрос: {question}
    Ответ:
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Ошибка генерации с контекстом: {e}")
        return f"Ошибка: {str(e)}"

async def generate_with_context_yandex(question: str, context: str) -> str:
    """Генерация ответа с контекстом через Yandex GPT"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return "Yandex GPT не настроен"

    prompt = f"""
    Ты — медицинский ассистент. Проанализируй информацию из источников и ответь на вопрос.
    
    Информация из источников:
    {context}
    
    Вопрос: {question}
    Ответ:
    """

    try:
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
                        'temperature': 0.7,
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
        logger.error(f"Ошибка генерации с контекстом Yandex: {e}")
        return f"Ошибка: {str(e)}"

def search_open_sources(question: str) -> List[dict]:
    """Поиск по открытым медицинским источникам"""
    sources = []

    # Здесь можно интегрировать реальные API или веб-скрапинг
    # Пока возвращаем демонстрационные данные

    sources.append({
        "name": "MedlinePlus (NIH)",
        "url": "https://medlineplus.gov",
        "content": "Общая медицинская информация от Национальных институтов здоровья США. Источник авторитетной информации о заболеваниях, лекарствах и здоровье."
    })

    sources.append({
        "name": "WHO - Всемирная организация здравоохранения",
        "url": "https://who.int",
        "content": "Глобальные медицинские рекомендации и статистика по заболеваниям. Информация о пандемиях, вакцинации и профилактике."
    })

    sources.append({
        "name": "PubMed Central",
        "url": "https://ncbi.nlm.nih.gov/pmc",
        "content": "Бесплатная коллекция медицинских и биологических статей. Научные публикации по всем аспектам медицины."
    })

    sources.append({
        "name": "Cochrane Library",
        "url": "https://cochranelibrary.com",
        "content": "Систематические обзоры медицинских исследований. Доказательная медицина и мета-анализы клинических испытаний."
    })

    sources.append({
        "name": "CDC - Центры по контролю и профилактике заболеваний США",
        "url": "https://cdc.gov",
        "content": "Информация о профилактике заболеваний, эпидемиологии, вакцинации и общественном здоровье."
    })

    return sources

# Новые вспомогательные функции для open_sources режима
async def generate_with_gemini_open_sources(question: str, context: str) -> str:
    """Генерация ответа с помощью Google Gemini для открытых источников"""
    if not gemini_model:
        return "Модель Google Gemini недоступна"

    try:
        prompt = f"""
        Ты — медицинский ассистент. Проанализируй информацию из открытых источников и ответь на вопрос.
        
        Информация из открытых источников:
        {context}
        
        Вопрос: {question}
        Ответ (только на основе предоставленной информации):
        """.strip()

        logger.debug("💬 Генерация ответа с помощью Google Gemini (open sources)...")
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        logger.debug("✅ Ответ от Google Gemini (open sources) сгенерирован.")
        return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Google Gemini (open sources): {e}")
        return f"Ошибка генерации Google Gemini: {str(e)}"

async def generate_with_yandex_open_sources(question: str, context: str) -> str:
    """Генерация ответа с помощью Yandex GPT для открытых источников"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return "Yandex GPT не настроен"

    try:
        prompt = f"""
        Ты — медицинский ассистент. Проанализируй информацию из открытых источников и ответь на вопрос.
        
        Информация из открытых источников:
        {context}
        
        Вопрос: {question}
        Ответ (только на основе предоставленной информации):
        """.strip()

        logger.debug("💬 Генерация ответа с помощью Yandex GPT (open sources)...")
        async with httpx.AsyncClient() as client:
            yandex_response = await client.post(
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
            yandex_response.raise_for_status()
            yandex_data = yandex_response.json()
            answer = yandex_data['result']['alternatives'][0]['message']['text'].strip()
            logger.debug("✅ Ответ от Yandex GPT (open sources) сгенерирован.")
            return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Yandex GPT (open sources): {e}")
        return f"Ошибка генерации Yandex GPT: {str(e)}"

# Новые вспомогательные функции для complete_analysis
async def generate_with_gemini_complete(question: str, context: str) -> str:
    """Генерация ответа с помощью Google Gemini для полного анализа"""
    if not gemini_model:
        return "Модель Google Gemini недоступна"

    try:
        prompt = f"""
        АНАЛИЗИРУЙ СЛЕДУЮЩУЮ ИНФОРМАЦИЮ И ПРЕДОСТАВЬ КРАТКИЙ, НО ПОЛНЫЙ ОТВЕТ:
        
        ВОПРОС: {question}
        
        КОНТЕКСТ:
        {context}
        
        Предоставь профессиональный медицинский анализ. Будь точным и лаконичным.
        """

        logger.debug("💬 Генерация ответа с помощью Google Gemini (complete)...")
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        logger.debug("✅ Ответ от Google Gemini (complete) сгенерирован.")
        return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Google Gemini (complete): {e}")
        return f"Ошибка генерации Google Gemini: {str(e)}"

async def generate_with_yandex_complete(question: str, context: str) -> str:
    """Генерация ответа с помощью Yandex GPT для полного анализа"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        return "Yandex GPT не настроен"

    try:
        prompt = f"""
        АНАЛИЗИРУЙ СЛЕДУЮЩУЮ ИНФОРМАЦИЮ И ПРЕДОСТАВЬ КРАТКИЙ, НО ПОЛНЫЙ ОТВЕТ:
        
        ВОПРОС: {question}
        
        КОНТЕКСТ:
        {context}
        
        Предоставь профессиональный медицинский анализ. Будь точным и лаконичным.
        """

        logger.debug("💬 Генерация ответа с помощью Yandex GPT (complete)...")
        async with httpx.AsyncClient() as client:
            yandex_response = await client.post(
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
                        {'role': 'system', 'text': 'Ты — медицинский эксперт. Предоставляй точные, научно обоснованные ответы.'},
                        {'role': 'user', 'text': prompt}
                    ]
                },
                timeout=60.0
            )
            yandex_response.raise_for_status()
            yandex_data = yandex_response.json()
            answer = yandex_data['result']['alternatives'][0]['message']['text'].strip()
            logger.debug("✅ Ответ от Yandex GPT (complete) сгенерирован.")
            return answer

    except Exception as e:
        logger.error(f"❌ Ошибка генерации Yandex GPT (complete): {e}")
        return f"Ошибка генерации Yandex GPT: {str(e)}"

# --- Эндпоинты API ---
@app.get("/")
async def home():
    """Корневой эндпоинт для проверки состояния сервиса."""
    try:
        vector_count = 0
        dimension = 'N/A'
        if pinecone_index:
            try:
                stats = pinecone_index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                dimension = stats.get('dimension', 'N/A')
            except Exception as e:
                logger.warning(f"Не удалось получить статистику Pinecone в /: {e}")
                vector_count = f"Ошибка: {e}"
        else:
            vector_count = "Не инициализирован"

        return {
            "status": "ok",
            "message": "Medical RAG API Server (Pinecone Llama Embeddings + Vertex AI)",
            "version": "3.4.0-Pinecone-Llama-Only",
            "pinecone_vectors": vector_count,
            "vector_dimension": dimension,
            "pinecone_embedding_model": "llama-text-embed-v2",
            "models_initialized": {
                "gemini_model": gemini_model is not None,
                "pinecone_index": pinecone_index is not None,
                "pinecone_client": pinecone_client is not None
            },
            "initialization_error": INITIALIZATION_ERROR
        }
    except Exception as e:
        logger.error(f"Ошибка в корневом эндпоинте: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса."""
    try:
        # Проверяем Pinecone
        pinecone_status = "uninitialized"
        vector_count = 0
        dimension = 'N/A'
        if pinecone_index is not None:
            try:
                stats = pinecone_index.describe_index_stats()
                pinecone_status = "healthy"
                vector_count = stats.get('total_vector_count', 0)
                dimension = stats.get('dimension', 'N/A')
            except Exception as e:
                pinecone_status = "unhealthy"
                logger.error(f"Health check - Pinecone error: {e}")

        # Проверяем модели
        gemini_model_status = "healthy" if gemini_model is not None else "uninitialized"
        pinecone_client_status = "healthy" if pinecone_client is not None else "uninitialized"
        pinecone_index_status = "healthy" if pinecone_index is not None else "uninitialized"

        overall_status = "healthy"
        if not all([
            pinecone_status == "healthy",
            gemini_model_status == "healthy",
            pinecone_client_status == "healthy",
            pinecone_index_status == "healthy"
        ]):
            if INITIALIZATION_ERROR:
                overall_status = "unhealthy"
            else:
                overall_status = "degraded"

        return {
            "status": overall_status,
            "components": {
                "pinecone": {
                    "status": pinecone_status,
                    "vector_count": vector_count,
                    "dimension": dimension,
                    "embedding_model": "llama-text-embed-v2"
                },
                "gemini_model": {
                    "status": gemini_model_status
                },
                "pinecone_client": {
                    "status": pinecone_client_status
                },
                "pinecone_index": {
                    "status": pinecone_index_status
                }
            },
            "initialization_error": INITIALIZATION_ERROR,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Ошибка в /health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Эндпоинт для ответа на медицинские вопросы с использованием RAG.
    """
    question = request.question.strip()
    mode = request.mode
    logger.info(f"📥 Получен вопрос: {question} (режим: {mode})")

    # 1. Валидация входных данных
    if not question:
        logger.warning("Получен пустой вопрос")
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    if len(question) > 1000:
        logger.warning(f"Получен слишком длинный вопрос ({len(question)} символов)")
        raise HTTPException(status_code=400, detail="Вопрос слишком длинный (максимум 1000 символов)")

    # 2. Проверка инициализации компонентов
    if not gemini_model and not (YANDEX_API_KEY and YANDEX_FOLDER_ID):
        error_msg = "Сервис не готов: ни одна модель генерации не доступна"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    if not pinecone_index or not pinecone_client:
        error_msg = "Сервис не готов: база знаний недоступна"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        # 3. Обработка в зависимости от режима
        answer = ""
        sources = []

        if mode == "knowledge_base":
            # Только поиск по базе знаний
            contexts, sources = await search_knowledge_base(question, top_k=3)
            answer = "Найденная информация:\n\n" + "\n\n---\n\n".join(contexts)

        elif mode == "combined_ai":
            # Поиск + генерация ответов от ИИ + объединенный ответ
            contexts, sources = await search_knowledge_base(question, top_k=3)
            logger.info(f"🔍 Найдено {len(contexts)} релевантных документов")

            if not contexts or contexts == ["База знаний недоступна"]:
                answer = "База знаний недоступна"
                sources = ["Система"]
            else:
                # Генерация ответов от разных моделей
                tasks = []
                models_used = []

                if gemini_model:
                    tasks.append(generate_with_gemini(question, contexts))
                    models_used.append("Google Gemini")
                if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                    tasks.append(generate_with_yandex(question, contexts))
                    models_used.append("Yandex GPT")

                if tasks:
                    answers = await asyncio.gather(*tasks, return_exceptions=True)

                    # Формирование ответов от каждой модели
                    model_answers = []
                    for model_name, ans in zip(models_used, answers):
                        if isinstance(ans, Exception):
                            logger.error(f"Ошибка генерации для {model_name}: {ans}")
                            model_answers.append(f"❌ Ошибка от {model_name}")
                        else:
                            model_answers.append(f"✅ Ответ от {model_name}:\n{ans}")

                    # Генерация объединенного ответа
                    unified_prompt = f"""
                    Вопрос: {question}
                    
                    Ответы от разных моделей:
                    {'\n\n'.join(model_answers)}
                    
                    Пожалуйста, объедините информацию из всех ответов в один логически связный и точный ответ.
                    Устраните противоречия, если они есть, и представьте наиболее полную информацию.
                    """

                    unified_answer = "Не удалось сгенерировать объединенный ответ"
                    if gemini_model:
                        try:
                            unified_response = gemini_model.generate_content(unified_prompt)
                            unified_answer = unified_response.text.strip()
                        except Exception as e:
                            logger.error(f"Ошибка генерации объединенного ответа: {e}")

                    # Формирование финального ответа
                    answer_parts = model_answers.copy()
                    answer_parts.append(f"📊 Объединенный анализ:\n{unified_answer}")
                    answer = "\n\n---\n\n".join(answer_parts)
                else:
                    answer = "Ни одна модель ИИ не доступна"
                    sources = ["Система"]

        elif mode == "unified_ai":
            # Объединенный ответ от ИИ без использования базы знаний
            answer = "🧠 Генерация ответа без использования базы знаний...\n\n"

            # Создаем пустой контекст для генерации без базы знаний
            empty_contexts = []

            tasks = []
            models_used = []

            if gemini_model:
                tasks.append(generate_with_gemini(question, empty_contexts))
                models_used.append("Google Gemini")
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_with_yandex(question, empty_contexts))
                models_used.append("Yandex GPT")

            if tasks:
                answers = await asyncio.gather(*tasks, return_exceptions=True)

                # Формирование ответов от каждой модели
                model_answers = []
                for model_name, ans in zip(models_used, answers):
                    if isinstance(ans, Exception):
                        logger.error(f"Ошибка генерации для {model_name}: {ans}")
                        model_answers.append(f"❌ Ошибка от {model_name}")
                    else:
                        model_answers.append(f"✅ Ответ от {model_name}:\n{ans}")

                # Генерация объединенного ответа
                unified_prompt = f"""
                Вопрос: {question}
                
                Ответы от разных моделей:
                {'\n\n'.join(model_answers)}
                
                Пожалуйста, объедините информацию из всех ответов в один логически связный и точный ответ.
                Устраните противоречия, если они есть, и представьте наиболее полную информацию.
                """

                unified_answer = "Не удалось сгенерировать объединенный ответ"
                if gemini_model:
                    try:
                        unified_response = gemini_model.generate_content(unified_prompt)
                        unified_answer = unified_response.text.strip()
                    except Exception as e:
                        logger.error(f"Ошибка генерации объединенного ответа: {e}")

                # Формирование финального ответа
                answer_parts = model_answers.copy()
                answer_parts.append(f"📊 Объединенный анализ:\n{unified_answer}")
                answer += "\n\n---\n\n".join(answer_parts)
                sources = ["Google Gemini", "Yandex GPT", "Объединенный анализ"]
            else:
                answer = "Ни одна модель ИИ не доступна"
                sources = ["Система"]

        elif mode == "open_sources":
            # Поиск по открытым источникам + объединенный ответ ИИ
            open_sources_info = search_open_sources(question)
            sources = [src["name"] for src in open_sources_info]

            # Формирование контекста из открытых источников
            open_contexts = []
            for src in open_sources_info:
                open_contexts.append(f"Источник: {src['name']}\n{src['content']}")

            context_text = "\n\n".join(open_contexts) if open_contexts else "Информация не найдена"

            # Генерация ответов от ИИ на основе открытых источников
            tasks = []
            models_used = []

            if gemini_model:
                tasks.append(generate_with_gemini_open_sources(question, context_text))
                models_used.append("Google Gemini")
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_with_yandex_open_sources(question, context_text))
                models_used.append("Yandex GPT")

            if tasks:
                answers = await asyncio.gather(*tasks, return_exceptions=True)

                # Формирование ответов от каждой модели
                model_answers = []
                for model_name, ans in zip(models_used, answers):
                    if isinstance(ans, Exception):
                        logger.error(f"Ошибка генерации для {model_name}: {ans}")
                        model_answers.append(f"❌ Ошибка от {model_name}")
                    else:
                        model_answers.append(f"✅ Ответ от {model_name}:\n{ans}")

                # Генерация объединенного ответа
                unified_prompt = f"""
                Вопрос: {question}
                
                Информация из открытых источников:
                {context_text}
                
                Ответы от разных моделей:
                {'\n\n'.join(model_answers)}
                
                Пожалуйста, объедините информацию из всех ответов в один логически связный и точный ответ.
                Устраните противоречия, если они есть, и представьте наиболее полную информацию.
                """

                unified_answer = "Не удалось сгенерировать объединенный ответ"
                if gemini_model:
                    try:
                        unified_response = gemini_model.generate_content(unified_prompt)
                        unified_answer = unified_response.text.strip()
                    except Exception as e:
                        logger.error(f"Ошибка генерации объединенного ответа: {e}")

                # Формирование финального ответа
                answer_parts = [
                    f"📚 Информация из открытых источников:\n{context_text}",
                    *model_answers,
                    f"📊 Объединенный анализ:\n{unified_answer}"
                ]
                answer = "\n\n---\n\n".join(answer_parts)
            else:
                answer = f"📚 Информация из открытых источников:\n{context_text}\n\nНи одна модель ИИ не доступна для анализа"

        elif mode == "complete_analysis":
            # ПОЛНЫЙ АНАЛИЗ: база знаний + открытые источники + ИИ
            logger.info("🚀 Начало полного анализа...")

            # 1. Получаем контент из базы знаний
            kb_contexts, kb_sources = await search_knowledge_base(question, top_k=3)
            logger.info(f"   База знаний: найдено {len(kb_contexts)} документов")

            # 2. Получаем информацию из открытых источников
            open_sources_info = search_open_sources(question)
            open_sources = [src["name"] for src in open_sources_info]
            logger.info(f"   Открытые источники: найдено {len(open_sources_info)} источников")

            # 3. Формируем полный контекст
            full_context_parts = []

            # Контекст из базы знаний
            if kb_contexts and kb_contexts != ["База знаний недоступна"]:
                kb_text = "\n\n".join(kb_contexts)
                full_context_parts.append(f"ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ:\n{kb_text}")

            # Контекст из открытых источников
            if open_sources_info:
                open_contexts = []
                for src in open_sources_info:
                    open_contexts.append(f"Источник: {src['name']}\n{src['content']}")
                open_text = "\n\n".join(open_contexts)
                full_context_parts.append(f"ИНФОРМАЦИЯ ИЗ ОТКРЫТЫХ ИСТОЧНИКОВ:\n{open_text}")

            full_context = "\n\n" + "\n\n" + "="*50 + "\n\n".join(full_context_parts) + "\n" + "="*50

            # 4. Генерируем ответы от ИИ
            ai_answers = []
            models_used = []

            # Google Gemini
            if gemini_model:
                try:
                    gemini_answer = await generate_with_gemini_complete(question, full_context)
                    ai_answers.append(f"ОТВЕТ ОТ GOOGLE GEMINI:\n{gemini_answer}")
                    models_used.append("Google Gemini")
                    logger.info("   ✅ Google Gemini: ответ сгенерирован")
                except Exception as e:
                    logger.error(f"   ❌ Google Gemini ошибка: {e}")
                    ai_answers.append(f"ОТВЕТ ОТ GOOGLE GEMINI:\nОшибка генерации")

            # Yandex GPT
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                try:
                    yandex_answer = await generate_with_yandex_complete(question, full_context)
                    ai_answers.append(f"ОТВЕТ ОТ YANDEX GPT:\n{yandex_answer}")
                    models_used.append("Yandex GPT")
                    logger.info("   ✅ Yandex GPT: ответ сгенерирован")
                except Exception as e:
                    logger.error(f"   ❌ Yandex GPT ошибка: {e}")
                    ai_answers.append(f"ОТВЕТ ОТ YANDEX GPT:\nОшибка генерации")

            # 5. Создаем финальный объединенный ответ
            if ai_answers:
                final_prompt = f"""
                ВЫПОЛНИТЕ ПОЛНЫЙ МЕДИЦИНСКИЙ АНАЛИЗ СЛЕДУЮЩЕГО ВОПРОСА:
                
                ВОПРОС: {question}
                
                ПОЛУЧЕННАЯ ИНФОРМАЦИЯ:
                {full_context}
                
                АНАЛИЗЫ ОТ СИСТЕМ ИИ:
                {'\n\n'.join(ai_answers)}
                
                ЗАДАЧА:
                Создайте научно обоснованный, структурированный ответ на медицинский вопрос.
                Используйте только достоверную информацию из предоставленных источников.
                Ответ должен быть профессиональным, но понятным.
                
                СТРУКТУРА ОТВЕТА:
                1. КЛИНИЧЕСКИЕ ПРОЯВЛЕНИЯ
                2. ДИАГНОСТИЧЕСКИЕ КРИТЕРИИ
                3. КЛИНИЧЕСКАЯ ЗНАЧИМОСТЬ
                4. РЕКОМЕНДАЦИИ
                5. ИСТОЧНИКИ ИНФОРМАЦИИ
                
                ВАЖНО: Если информации недостаточно, честно это укажите.
                НЕ ВЫДУМЫВАЙТЕ информацию. Используйте только предоставленные данные.
                """

                try:
                    if gemini_model:
                        final_response = gemini_model.generate_content(final_prompt)
                        final_answer = final_response.text.strip()
                        logger.info("   ✅ Финальный ответ сгенерирован")
                    else:
                        final_answer = "Финальный ответ не может быть сгенерирован: модель недоступна"
                except Exception as e:
                    logger.error(f"   ❌ Ошибка генерации финального ответа: {e}")
                    final_answer = f"Ошибка генерации финального ответа: {str(e)}"
            else:
                final_answer = "Не удалось получить анализы от систем ИИ"

            # 6. Формируем финальный ответ
            answer = f"ПОЛНЫЙ МЕДИЦИНСКИЙ АНАЛИЗ\n\n{final_answer}"

            # 7. Собираем все источники
            all_sources = []
            if kb_sources and kb_sources != ["Система"]:
                all_sources.extend(kb_sources)
            all_sources.extend(open_sources)
            all_sources.extend(models_used)
            sources = list(set(all_sources))  # Убираем дубликаты

            logger.info("✅ Полный анализ завершен")

        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый режим")

        # 5. Возвращаем ответ
        logger.info("✅ Ответ сгенерирован и отправлен")
        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
            mode=mode
        )

    except HTTPException:
        # Перебрасываем HTTPException как есть
        raise
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка в /ask: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    from datetime import datetime
    # Получаем порт из переменной окружения, установленной Cloud Run/Render
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Запуск сервера Uvicorn на порту {port}...")
    # ВАЖНО: host должен быть "0.0.0.0" для Cloud Run/Render
    uvicorn.run(app, host="0.0.0.0", port=port)
