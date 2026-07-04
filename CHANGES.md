# CHANGES.md - Лог изменений

Документ описывает все внесённые изменения для понимания другими агентами/разработчиками.

---

## Рефакторинг app.py

### Устранение дублирования `_build_command()`

**Было:** `LlamaServer._build_command()` (строки 261-295) и `ModelPool._build_command()` (строки 360-396) содержали ~30 строк идентичного кода. Разница только в источнике port/ctx_size/gpu_layers.

**Стало:** Вынесена общая функция `build_llama_command()` (строки ~87-118), используемая обоими классами. Добавлена вспомогательная `_normalize_flash_attn()` для нормализации значений flash attention.

### Устранение дублирования HTTP-пейлоада чата

**Было:** Конструирование JSON-пейлоада для llama-server дублировалось в `LlamaServer.chat()` и в `api_chat()`.

**Стало:** Вынесена общая функция `build_chat_payload(messages, settings, stream)`. Ключевое изменение: `max_tokens` теперь **не отправляется** в payload если значение `-1` (безлимит), что предотвращает возможные проблемы с llama-server.

### Очистка singleton-сервера при остановке

**Было:** `LlamaServer.stop()` не очищал `process`, `output_queue`, `logs` после остановки.

**Стало:** `LlamaServer.stop()` теперь очищает все ресурсы аналогично `ModelPool.stop_slot()`:
- Очистка output_queue
- Очистка logs
- Установка process = None

### Добавлен метод `restart()`

Добавлен метод `LlamaServer.restart()` — останавливает и запускает сервер заново. Используется новым эндпоинтом `/api/server/restart`.

---

## Безопасность

### Перенос API-ключа в переменные окружения

**Было:** `orchestrator_api_key` хранился в открытом `runner_config.json`.

**Стало:** Приоритет чтения ключа:
1. `os.environ.get("ORCHESTRATOR_API_KEY")` (первая проверка)
2. `config.settings["orchestrator_api_key"]` (fallback)

Изменения в `orchestrator_api.py`: `check_api_key()` теперь сначала проверяет env-переменную.

### .gitignore

Добавлен `runner_config.json` в `.gitignore` — файл с секретами больше не будет закоммичен.

### .env.example

Создан `.env.example` с комментарием по настройке `ORCHESTRATOR_API_KEY`.

---

## Новые API-эндпоинты

### `GET /api/pool/models`

Возвращает JSON с назначенными моделями для каждого слота:
```json
{"assigned": {"fast": "path/to/model.gguf", "quality": "path/to/other.gguf"}}
```

Используется фронтендом для кросстаб-подсветки моделей.

### `POST /api/server/restart`

Принимает `{settings: {...}, selected_model: "..."}`. Сохраняет настройки, останавливает текущий процесс, запускает заново.

---

## Фронтенд (templates/index.html)

### Кросстаб-подсветка моделей

**Проблема:** Модели, назначенные в пул слотов, не отображались в боковой панели моделей.

**Решение:**
- Добавлена глобальная переменная `poolAssignedModels`
- Функция `loadPoolAssignedModels()` загружает данные с `GET /api/pool/models`
- В `renderModelsList()` добавлены цветные бейджи `ПУЛ: FAST`, `ПУЛ: QUALITY` и т.д. для моделей, назначенных в слоты
- Бейджи цветовые: FAST=синий, QUALITY=фиолетовый, MEDIUM=янтарный, HEAVY=оранжевый
- При назначении модели в слот (`poolAssignSlot`) — автоматически обновляется подсветка в сайдбаре

### Автоперезагрузка сервера

**Проблема:** При изменении настроек (ctx_size, gpu_layers и пр.) модель не перезагружалась.

**Решение:**
- Добавлена кнопка «Применить и перезапустить» в футере модалки настроек
- Функция `saveAndRestart()`: сохраняет настройки + вызывает `POST /api/server/restart`
- Кнопка «Сохранить» сохраняет без перезапуска (как раньше)

### Исправление max_tokens

**Проблема:** `max_tokens: 1024` в config обрывал ответы модели.

**Решение:**
- В `build_chat_payload()` (backend): `max_tokens` не включается в payload если значение `-1`
- В UI: добавлена подсказка `-1 = без ограничений (до конца ответа)`

---

## CUDA / GPU / P100 поддержка

### Автоматическое определение GPU

Добавлен модуль определения GPU в `app.py`:

- **`detect_gpu()`** — определяет NVIDIA GPU через `nvidia-smi`: имя, VRAM (общая/свободная), compute capability, архитектуру, версию драйвера
- **`detect_cuda_version()`** — определяет версию CUDA runtime, проверяет наличие DLL в папке llama.cpp
- **`get_recommended_settings(gpu)`** — возвращает рекомендованные настройки для конкретного GPU

### Таблица архитектур GPU

Добавлена константа `GPU_ARCH_INFO` с данными для всех архитектур NVIDIA:
- Maxwell (CC 5.0-5.2): flash_attn=нет, max_ctx=8192
- **Pascal (CC 6.0-6.1): flash_attn=нет, max_ctx=16384** ← P100, GTX 10xx
- Volta (CC 7.0): flash_attn=да, max_ctx=32768
- Turing (CC 7.5): flash_attn=да, max_ctx=32768
- Ampere (CC 8.0-8.6): flash_attn=да, max_ctx=65536
- Ada (CC 8.9): flash_attn=да, max_ctx=65536
- Hopper (CC 9.0): flash_attn=да, max_ctx=131072
- Blackwell (CC 10.0-12.0): flash_attn=да, max_ctx=131072

### NVIDIA P100 специфика

При обнаружении P100 автоматически:
- **Flash Attention отключен** (Pascal не поддерживает)
- **KV Cache тип: f16** вместо q4_0 (экономия VRAM другим способом)
- **Рекомендуемый контекст: ≤16384** (ограничение пропускной способности HBM2)
- **GPU слои: auto** (llama.cpp сам оптимизирует)
- В UI показывается предупреждение: «⚠ NVIDIA P100 обнаружен»

### API-эндпоинты

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/gpu/info` | GET | Полная информация: GPU, CUDA, рекомендации |
| `/api/gpu/recommend` | POST | Рекомендации для конкретного GPU по индексу |
| `/api/gpu/apply-recommended` | POST | Применить рекомендованные настройки |
| `/api/cuda/update` | POST | Проверить и скопировать отсутствующие CUDA DLLs |

### Авто-обновление CUDA DLLs

Эндпоинт `/api/cuda/update`:
1. Проверяет какие DLL отсутствуют в папке llama.cpp
2. Пытается найти их в CUDA Toolkit (CUDA_PATH)
3. Копирует найденные DLL
4. Возвращает список скопированных/отсутствующих файлов

### Фронтенд — вкладка GPU / CUDA

В модалке настроек добавлена вкладка **GPU / CUDA** с тремя секциями:

1. **Обнаруженное GPU** — имя, архитектура, VRAM, Flash Attn поддержка
2. **CUDA Runtime** — версия runtime, статус DLL, кнопка обновления
3. **Рекомендуемые настройки** — что будет применено, кнопка «Применить»

---

## User Profiles — личность пользователя (`llm_orchestrator/user_profiles.py`)

### Концепция

При первом обращении система запрашивает у пользователя информацию о себе:
- Имя, должность, отдел
- Частые задачи через ERP
- Правила и регламенты
- Предпочтения в формате ответов

Эта информация сохраняется в `user_profiles/{id}.md` и используется как
контекст для всех будущих сессий этого пользователя.

### Как работает

1. **Первое обращение** → `onboarded=False` → агент получает промпт с вопросами
2. **Пользователь отвечает** → извлекаются имя, роль, отдел, правила
3. **При последующих запросах** → в system prompt добавляется контекст пользователя
4. **Паттерны запросов** автоматически извлекаются и записываются

### API

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/orchestrator/profile` | GET | Получить профиль пользователя |
| `/api/orchestrator/profile` | POST | Обновить профиль |
| `/api/orchestrator/profile/onboard` | POST | Завершить онбординг |

### Формат файла `user_profiles/{id}.md`

```markdown
# UserProfile: ivanov

**Имя:** Иван Иванов
**Роль:** Инженер ПНР
**Отдел:** ПРС
**Онбординг:** Да

## Правила
- При обновлении нарядов всегда указывать автора
- Даты проверять дважды

## Предпочтения
- формат: коротко, с таблицами

## Паттерны запросов
- Управление задачами
- Работа с проектами
```

---

## Verification Engine — watchdog-паттерн (`llm_orchestrator/verification.py`)

### Концепция

После каждого tool call агент автоматически проверяет результат через
другой инструмент. Если проверка не пройдена — повторяет или эскалирует.

### Паттерн верификации

```
Tool call (append_task_details) → OK
    ↓
Verifier (get_task_comments) → проверяем, что запись появилась
    ↓
PASS → продолжаем
FAIL → повторяем с другим подходом
```

### Правила верификации

| Инструмент | Верификация через | Порог confidence |
|------------|-------------------|------------------|
| `append_task_details` | `get_task_comments` | 0.8 |
| `update_task_summary` | `get_task_comments` | 0.8 |
| `consolidate_to_project` | `get_project_card` | 0.7 |
| `get_project_card` | (чтение — OK) | 0.9 |
| `execute_python` | (stdout check) | 0.7 |

### API

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/orchestrator/verification/stats` | GET | Статистика верификаций |

### Confidence scoring

Confidence рассчитывается на основе:
- ОК от основного инструмента (+0.2)
- Наличия данных в верификации (+0.15)
- Заполненности целевого поля (+0.1)
- Штраф за пустой результат (-0.2)

---

## Agent Philosophy — эволюция принципов (`llm_orchestrator/philosophy.py`)

### Концепция

Глобальный файл `philosophy.md` определяет «характер» агента:
- Миссия и ключевые принципы
- Правила работы с ERP
- Чего НЕ делать
- Эволюция: успешные паттерны и ошибки добавляются автоматически

### Автоматическая эволюция

- **Успех** → `record_success(task_type, approach)` → добавляется в principles
- **Ошибка** → `record_failure(task_type, error, fix)` → добавляется в anti_patterns
- Файл растёт по мере работы агента

### API

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/orchestrator/philosophy` | GET | Текущая философия |
| `/api/orchestrator/philosophy` | POST | Обновить философию |

---

## Интеграция в agentic loop

В `run_orchestrator()` добавлены:

1. **User context injection** — профиль пользователя + философия добавляются в промпт
2. **Onboarding detection** — первый запрос пользователя → вопросы об интерфейсе
3. **Post-action verification** — после каждого tool call → автоматическая проверка
4. **Interaction recording** — каждый запрос записывается в профиль пользователя
5. **Philosophy evolution** — успешные/неуспешные действия обновляют принципы

---

## Multi-Client платформа (`llm_orchestrator/client_registry.py`, `client_tools.py`)

### Концепция

ERP (L-Start) — один из нескольких клиентов AI-агента. Каждый клиент:
- Подключается через `X-Client-ID` + `X-Client-API-Key`
- Имеет свой набор инструментов, навыков и system prompt
- Изолирован от других клиентов

### Архитектура

```
Клиент A (ERP)    → X-Client-ID: erp     → ToolRegistry_A + Skills_A + Prompt_A
Клиент B (CRM)    → X-Client-ID: crm     → ToolRegistry_B + Skills_B + Prompt_B
Клиент C (Telegram) → X-Client-ID: tg_bot → ToolRegistry_C + Skills_C + Prompt_C
```

### Идентификация

- **Multi-client mode**: заголовки `X-Client-ID` + `X-Client-API-Key`
- **Legacy mode**: заголовок `X-Orchestrator-API-Key` (обратная совместимость)
- Без заголовков на localhost → доступ разрешён

### Файлы

| Файл | Описание |
|------|----------|
| `client_registry.py` | ClientConfig dataclass, CRUD, валидация API ключей |
| `client_tools.py` | Фабрика ToolRegistry для клиента (базовые + кастомные инструменты) |
| `clients/*.json` | Конфиги клиентов (хранилище) |

### API управления клиентами

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/clients` | GET | Список клиентов |
| `/api/clients` | POST | Создать клиента |
| `/api/clients/<id>` | GET | Получить клиента |
| `/api/clients/<id>` | PUT | Обновить клиента |
| `/api/clients/<id>` | DELETE | Удалить клиента |
| `/api/clients/<id>/tools` | GET | Инструменты клиента |
| `/api/clients/<id>/tools` | POST | Добавить инструмент |
| `/api/clients/<id>/tools/<name>` | DELETE | Удалить инструмент |

### Формат конфига клиента

```json
{
  "client_id": "erp",
  "name": "ERP L-Start",
  "api_key": "erp_secret_key_2026",
  "enabled": true,
  "system_prompt": "Ты ИИ-ассистент ERP-системы L-Start...",
  "tools": [...],
  "skills": {},
  "allowed_models": [],
  "rate_limit": 10
}
```

### Изоляция инструментов

- **Базовые инструменты** (execute_python, monitor_web, search_kb) — доступны всем клиентам
- **ERP-инструменты** — только для клиента "erp"
- **Кастомные инструменты** — из конфига клиента

---

## File Analysis & Vision (`llm_orchestrator/file_tools.py`)

### Концепция

Агент получает возможность работать с файлами:
- Загружать изображения и текст через API
- Анализировать изображения через vision-модель (mmproj)
- Читать и анализировать текстовые файлы

### Vision API (OpenAI-совместимый)

Для анализа изображений используется mmproj-gemma-4-12B-it-BF16:

```python
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Что на картинке?"},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,<BASE64>"
            }}
        ]
    }],
    "max_tokens": 1000
}
```

### Инструменты агента

| Инструмент | Описание |
|------------|----------|
| `read_file(path)` | Чтение текстовых файлов (код, TXT, MD, JSON) |
| `analyze_image(path, question)` | Vision-анализ изображения через mmproj |
| `list_files(directory)` | Список файлов в директории |
| `get_file_info(path)` | Метаданные файла (размер, тип) |

### API загрузки файлов

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/files/upload` | POST | Загрузка файла (multipart form) |
| `/api/files/upload-base64` | POST | Загрузка через base64 |
| `/api/files/list` | GET | Список загруженных файлов |
| `/api/files/<name>` | GET | Скачивание файла |
| `/api/files/<name>` | DELETE | Удаление файла |

### Поддерживаемые форматы

**Текст:** .py, .js, .ts, .html, .css, .json, .yaml, .md, .txt, .sql, .sh, .c, .cpp, .go, .rs и др.
**Изображения:** .png, .jpg, .jpeg, .gif, .webp, .bmp, .tiff

### Ограничения

- Макс. размер текстового файла: 100 KB
- Макс. размер изображения: 20 MB
- Vision-анализ требует запущенный llama-server с mmproj

---

## llama.cpp Auto-Update

### Функция

Автоматическая проверка и обновление llama.cpp до последней версии с GitHub.

### API

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/llamacpp/version` | GET | Текущая и последняя версия |
| `/api/llamacpp/update` | POST | Скачать и установить последний релиз |

### Как работает

1. `check_llama_cpp_version()` — запускает `llama-server --version` для текущей версии
2. Запрашивает GitHub API для последнего релиза (`ggml-org/llama.cpp`)
3. Скачивает Windows ZIP-архив
4. Извлекает .exe и .dll файлы в папку llama.cpp

### UI

Вкладка «GPU / CUDA» → секция «llama.cpp»:
- Текущая версия
- Последняя версия
- Кнопка «Обновить»

---

## Vision Auto-Detection & Toggle

### Как работает

При сканировании моделей система автоматически ищет mmproj-файл рядом с основной моделью:

1. Ищет `*mmproj*.gguf` в той же папке
2. Сверяет имена (gemma-4-12B + mmproj-gemma-4-12B → совпадение)
3. Если найден — на карточке модели появляется **иконка глаза**

### Активация Vision

Нажатие на иконку глаза:
1. Спрашивает подтверждение
2. Устанавливает `mmproj` в конфиг
3. Выбирает эту модель как основную
4. Перезапускает сервер

### UI

- **Иконка глаза** — появляется при наведении на модель с mmproj
- **Бейдж VISION** — показывается если vision активен для модели
- **Фиолетовый цвет** — индикация активного vision

### API

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/models/vision` | POST | Включить/отключить vision для модели |

### Пример запроса

```json
POST /api/models/vision
{
  "model_path": "F:\\Models\\gemma-4-12B-it-Q6_K.gguf",
  "mmproj": "F:\\Models\\mmproj-gemma-4-12B-it-BF16.gguf"
}
```

---

## Очистка кода

- Убран повторный `from tools import ERPIntegrationTools` в `orchestrator_api.py`
- Убран неиспользуемый `RouterLLM` из импортов (оставлен только `classify_complexity`)
