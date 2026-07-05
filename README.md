# Llama Runner MIMO

Локальный загрузчик LLM-моделей с веб-интерфейсом и оркестратором агентов для ERP-системы L-Start.

## Возможности

- **Загрузка моделей** — запуск GGUF-моделей через llama-server.exe с настройкой параметров
- **Мульти-модельный пул** — одновременная загрузка нескольких моделей на разных портах
- **Роутинг по сложности** — автоматический выбор модели по сложности запроса
- **Оркестратор агентов** — Agentic loop с инструментами (ERP, Python sandbox, мониторинг)
- **Мульти-пользовательность** — аккаунты с ролями (admin/user/readonly)
- **Безопасность** — аутентификация, CORS, security headers, path traversal защита
- **WireGuard** — доступ через VPN без дополнительной аутентификации
- **Drag & Drop** — перетаскивание файлов в чат для анализа

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
python app.py
# или
run.bat
```

Открыть http://127.0.0.1:5000

## Аутентификация

### Веб-интерфейс
- Логин: `admin`
- Пароль: значение `orchestrator_api_key` из `runner_config.json`

### API
```bash
curl -H "X-Orchestrator-API-Key: hermes_secret_api_key_2026" \
  http://localhost:5000/api/server/status
```

### WireGuard
При подключении через VPN (10.10.0.x) аутентификация не требуется.

## API

Подробная документация: [API.md](API.md)

| Эндпоинт | Метод | Описание |
|-----------|-------|----------|
| `/api/auth/login` | POST | Вход в систему |
| `/api/accounts` | GET/POST | Управление аккаунтами |
| `/api/models` | GET | Список моделей |
| `/api/server/start` | POST | Запустить сервер |
| `/api/server/stop` | POST | Остановить сервер |
| `/api/server/status` | GET | Статус сервера |
| `/api/pool/status` | GET | Статус пула моделей |
| `/api/pool/start` | POST | Запустить все слоты |
| `/api/chat` | POST | Отправить сообщение |
| `/api/files/upload` | POST | Загрузить файл |
| `/api/orchestrator/run` | POST | Запустить агента |

## Конфигурация

### runner_config.json
```json
{
  "settings": {
    "host": "127.0.0.1",
    "port": 8080,
    "orchestrator_api_key": "your_secret_key",
    "allowed_origins": []
  }
}
```

### .env
```
ORCHESTRATOR_API_KEY=your_secret_key
ERP_API_KEY=erp_secret_key
FLASK_SECRET_KEY=auto_generated
```

## Архитектура

```
┌─────────────────────────────────────────────┐
│  Flask App (port 5000)                      │
│  ├── Web UI (templates/index.html)          │
│  ├── Account Manager (accounts.json)        │
│  ├── Model Pool                             │
│  │   ├── Slot "fast"    → llama-server :8080│
│  │   └── Slot "quality" → llama-server :8081│
│  └── Orchestrator API                       │
│      ├── Complexity Router                  │
│      ├── Tool Registry                      │
│      ├── Session Store (SQLite)             │
│      └── Self-Learner                       │
└─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
    llama-server.exe    ERP Backend (L-Start)
    (LLM inference)     (tools, knowledge)
```

## Стек

- Python 3.11+, Flask
- llama.cpp (llama-server.exe)
- SQLite (сессии, обучение)
- Vanilla JS + Tailwind CSS (фронтенд)
- WireGuard (VPN доступ)

## Безопасность

- Аутентификация: session-based (веб) + API keys (программно)
- Роли: admin, user, readonly
- Security headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- CORS: настраиваемый список разрешённых origin
- Path traversal: санитизация путей в файлах и чатах
- Command injection: list-form subprocess
- XSS: DOMPurify для markdown
- Upload limit: 100 MB
