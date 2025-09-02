# main.py - Medical RAG API с использованием встроенной модели Pinecone llama-text-embed-v2
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pinecone import Pinecone
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
import requests

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
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", 5000)) # Ограничение длины контекста

# --- Lifespan handler для FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model, pinecone_index, pinecone_client, INITIALIZATION_ERROR
    logger.info("🚀 Начало инициализации приложения Medical RAG API...")

    try:
        # 1. Инициализация Vertex AI (для генерации ответов)
        logger.info(f"🔧 Инициализация Vertex AI: project={PROJECT_ID}, location={REGION}")
        vertexai.init(project=PROJECT_ID, location=REGION)
        logger.info("✅ Vertex AI инициализирована.")

        # 2. Инициализация модели Google Gemini (для генерации ответов)
        logger.info("🧠 Загрузка модели генерации gemini-2.5-pro...")
        try:
            # Попробуем сначала 2.5-pro, если недоступна, используем 1.5-pro
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
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        # Подключаемся к индексу
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        index = pinecone_client.Index(PINECONE_INDEX_NAME)

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
    mode: str = "knowledge_base" # 'knowledge_base', 'combined_ai'

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    mode: str

# --- Вспомогательные функции ---
# main.py (на Render) - Исправленный фрагмент функции search_knowledge_base

# Убедитесь, что у вас есть глобальный объект клиента Pinecone
# Обычно создается в lifespan handler:
# pc = Pinecone(api_key=PINECONE_API_KEY)

async def search_knowledge_base(question: str, top_k: int = 3):
    """Асинхронный поиск по базе знаний Pinecone с использованием встроенной модели Llama."""
    global pinecone_index, pinecone_client

    if not pinecone_index:
        logger.warning("База знаний недоступна (индекс Pinecone не инициализирован)")
        return ["База знаний недоступна"], ["Система"]

    try:
        logger.debug(f"🔍 Поиск в Pinecone по запросу: '{question}'...")

        # --- 1. Создание эмбеддинга для вопроса с помощью Inference API Pinecone (асинхронно) ---
        logger.debug("🧠 Создание эмбеддинга для вопроса с помощью Pinecone Inference API...")

        # Оборачиваем блокирующий вызов в asyncio.to_thread
        embedding_response = await asyncio.to_thread(
            pinecone_client.inference.embed,
            model="llama-text-embed-v2",
            inputs=[question],
            parameters={
                "input_type": "query",
                "truncate": "END"
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
                text = metadata.get('content') or metadata.get('title') or metadata.get('preview')  or f"Документ ID: {match.id}"
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
        Ты — медицинский ассистент. Отвечай на вопрос, опираясь ТОЛЬКО на предоставленный контекст.
        Отвечай ясно, точно и по существу.
        Если ответа нет в контексте, скажи: "Я не могу дать медицинскую консультацию на основе предоставленных данных. Обратитесь к врачу."

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
        Ты — медицинский ассистент. Отвечай на вопрос, опираясь ТОЛЬКО на предоставленный контекст.
        Отвечай ясно, точно и по существу.
        Если ответа нет в контексте, скажи: "Я не могу дать медицинскую консультацию на основе предоставленных данных. Обратитесь к врачу."

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

    # 1. PubMed Central API (работает)
    try:
        logger.debug("Поиск в PubMed Central...")
        pmc_results = search_pubmed_central(question)
        if pmc_results:
            sources.extend(pmc_results)
    except Exception as e:
        logger.error(f"Ошибка поиска в PubMed Central: {e}")

    # 2. MedlinePlus API (работает)
    try:
        logger.debug("Поиск в MedlinePlus...")
        medline_results = search_medlineplus(question)
        if medline_results:
            sources.extend(medline_results)
    except Exception as e:
        logger.error(f"Ошибка поиска в MedlinePlus: {e}")

    # 3. WHO GHO API (работает)
    try:
        logger.debug("Поиск в WHO GHO...")
        who_results = search_who_gho(question)
        if who_results:
            sources.extend(who_results)
    except Exception as e:
        logger.error(f"Ошибка поиска в WHO GHO: {e}")

    # 4. Веб-сайты (через scraping - для демонстрации)
    try:
        logger.debug("Поиск в веб-источниках...")
        web_results = search_web_sources(question)
        if web_results:
            sources.extend(web_results)
    except Exception as e:
        logger.error(f"Ошибка поиска в веб-источниках: {e}")

    # Если ничего не найдено, возвращаем базовую информацию
    if not sources:
        sources.append({
            "name": "Общие медицинские источники",
            "url": "https://medlineplus.gov",
            "content": f"Для запроса '{question}' рекомендуется обратиться к авторитетным медицинским источникам:\n" +
                       "- MedlinePlus (NIH)\n" +
                       "- Всемирная организация здравоохранения (WHO)\n" +
                       "- PubMed Central\n" +
                       "- Центры по контролю и профилактике заболеваний (CDC)"
        })

    return sources[:5]  # Ограничиваем 5 результатами

def search_pubmed_central(query: str) -> List[dict]:
    """Поиск статей в PubMed Central"""
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pmc',
            'term': query,
            'retmax': 3,
            'format': 'json',
            'sort': 'relevance'
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        if 'idlist' in data['esearchresult']:
            pmc_ids = data['esearchresult']['idlist'][:2]  # Берем первые 2

            # Получаем детали статей
            for pmc_id in pmc_ids:
                try:
                    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    summary_params = {
                        'db': 'pmc',
                        'id': pmc_id,
                        'format': 'json'
                    }
                    summary_response = requests.get(summary_url, params=summary_params, timeout=5)
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                    if str(pmc_id) in summary_data['result']:
                        article_info = summary_data['result'][str(pmc_id)]
                        title = article_info.get('title', 'Без названия')
                        authors = article_info.get('authors', [])
                        author_names = [author.get('name', '') for author in authors[:3]]
                        authors_str = ', '.join(author_names) if author_names else 'Авторы не указаны'

                        results.append({
                            "name": f"PubMed Central: {title[:100]}...",
                            "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/",
                            "content": f"Научная статья по теме '{query}'. Заголовок: {title}. Авторы: {authors_str}"
                        })
                except Exception as e:
                    logger.warning(f"Ошибка получения деталей статьи PMC{pmc_id}: {e}")
                    continue

        return results
    except Exception as e:
        logger.error(f"Ошибка поиска в PubMed Central: {e}")
        return []

def search_medlineplus(query: str) -> List[dict]:
    """Поиск информации в MedlinePlus"""
    try:
        url = "https://wsearch.nlm.nih.gov/ws/query"
        params = {
            'db': 'healthTopics',
            'term': query,
            'retmax': 2
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        results = []
        if '<list>' in response.text:
            # Парсим XML ответ
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(response.text)
                items = root.findall('.//item')[:2]  # Берем первые 2

                for item in items:
                    title_elem = item.find('title')
                    url_elem = item.find('url')
                    title = title_elem.text if title_elem is not None else 'Без названия'
                    url = url_elem.text if url_elem is not None else 'https://medlineplus.gov'

                    results.append({
                        "name": f"MedlinePlus: {title[:100]}...",
                        "url": url,
                        "content": f"Медицинская информация по теме '{query}' от Национальных институтов здоровья США. Тема: {title}"
                    })
            except Exception as xml_error:
                logger.error(f"Ошибка парсинга XML MedlinePlus: {xml_error}")
                # Возвращаем общую информацию
                results.append({
                    "name": "MedlinePlus",
                    "url": "https://medlineplus.gov",
                    "content": f"Общая медицинская информация по запросу '{query}' доступна на MedlinePlus - авторитетном источнике NIH"
                })

        return results
    except Exception as e:
        logger.error(f"Ошибка поиска в MedlinePlus: {e}")
        return []

def search_who_gho(query: str) -> List[dict]:
    """Поиск статистики в WHO Global Health Observatory"""
    try:
        # Ищем подходящие индикаторы
        indicators_url = "https://ghoapi.azureedge.net/api/Indicator"
        indicators_response = requests.get(indicators_url, timeout=10)
        indicators_response.raise_for_status()
        indicators_data = indicators_response.json()

        results = []
        found_indicators = []

        # Ищем индикаторы, связанные с запросом
        if 'value' in indicators_data:
            for indicator in indicators_data['value'][:50]:  # Проверяем первые 50
                indicator_title = indicator.get('IndicatorName', '').lower()
                if query.lower() in indicator_title or any(word in indicator_title for word in query.lower().split()):
                    found_indicators.append(indicator)
                    if len(found_indicators) >= 2:
                        break

        # Получаем данные по найденным индикаторам
        for indicator in found_indicators[:2]:
            indicator_code = indicator.get('IndicatorCode', '')
            indicator_name = indicator.get('IndicatorName', 'Не указано')

            # Получаем последние данные
            data_url = f"https://ghoapi.azureedge.net/api/{indicator_code}?$top=3"
            try:
                data_response = requests.get(data_url, timeout=10)
                data_response.raise_for_status()
                data = data_response.json()

                if 'value' in data and len(data['value']) > 0:
                    sample_data = data['value'][0]
                    country = sample_data.get('SpatialDim', 'Не указано')
                    value = sample_data.get('NumericValue', 'Нет данных')
                    year = sample_data.get('TimeDim', 'Год не указан')

                    results.append({
                        "name": f"WHO GHO: {indicator_name[:80]}...",
                        "url": "https://www.who.int/data/gho",
                        "content": f"Глобальная статистика по '{indicator_name}'. Пример: {country}, {year} - {value}"
                    })
            except Exception as data_error:
                logger.warning(f"Ошибка получения данных по индикатору {indicator_code}: {data_error}")
                results.append({
                    "name": f"WHO GHO: {indicator_name[:80]}...",
                    "url": "https://www.who.int/data/gho",
                    "content": f"Глобальная статистика по '{indicator_name}' доступна в базе данных ВОЗ"
                })

        # Если не нашли специфических данных, возвращаем общую информацию
        if not results:
            results.append({
                "name": "WHO Global Health Observatory",
                "url": "https://www.who.int/data/gho",
                "content": f"Глобальная статистика здравоохранения по теме '{query}' доступна в базе данных Всемирной организации здравоохранения"
            })

        return results
    except Exception as e:
        logger.error(f"Ошибка поиска в WHO GHO: {e}")
        return []

def search_web_sources(query: str) -> List[dict]:
    """Поиск информации через веб-сайты (демонстрационная реализация)"""
    results = []

    # WHO сайт
    results.append({
        "name": "WHO - Всемирная организация здравоохранения",
        "url": "https://www.who.int",
        "content": f"Глобальные медицинские рекомендации и статистика по '{query}'. " +
                   "ВОЗ предоставляет авторитетную информацию о заболеваниях, профилактике и здравоохранении."
    })

    # CDC сайт
    results.append({
        "name": "CDC - Центры по контролю и профилактике заболеваний США",
        "url": "https://www.cdc.gov",
        "content": f"Информация о профилактике и лечении '{query}' от Центров по контролю и профилактике заболеваний США. " +
                   "Актуальные рекомендации и статистика по заболеваниям."
    })

    # NIH сайт
    results.append({
        "name": "NIH - Национальные институты здоровья США",
        "url": "https://www.nih.gov",
        "content": f"Научные исследования и медицинская информация по теме '{query}' от Национальных институтов здоровья США. " +
                   "Информация на основе последних научных данных."
    })

    return results[:2]  # Возвращаем максимум 2 результата

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
    Режимы:
    - knowledge_base: только поиск в базе знаний
    - combined_ai: поиск + генерация ответов от ИИ
    - unified_ai: объединенный ответ от ИИ без базы знаний
    - open_sources: поиск по открытым источникам + объединенный ответ ИИ
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

    try:
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
                            unified_answer = await generate_with_gemini(unified_prompt, [])
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

            tasks = []
            models_used = []

            if gemini_model:
                tasks.append(generate_without_context(question))
                models_used.append("Google Gemini")
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_without_context_yandex(question))
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
                tasks.append(generate_with_context(question, context_text))
                models_used.append("Google Gemini")
            if YANDEX_API_KEY and YANDEX_FOLDER_ID:
                tasks.append(generate_with_context_yandex(question, context_text))
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
        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый режим")

        logger.info("✅ Ответ сгенерирован и отправлен")
        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
            mode=mode
        )

    except HTTPException:
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
