# Llama Runner MIMO — API Integration Guide

## Базовый URL

```
http://<YOUR_IP>:5000
```

## Аутентификация

### Веб-интерфейс
- Логин: любой (по умолчанию `admin`)
- Пароль: значение `orchestrator_api_key`

### API-запросы
Заголовок в каждом запросе:
```
X-Orchestrator-API-Key: hermes_secret_api_key_2026
```

### Клиентская аутентификация (multi-client)
```
X-Client-ID: erp
X-Client-API-Key: erp_secret_key_2026
```

---

## Эндпоинты

### Аутентификация

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/auth/login` | Вход (username + password) |
| POST | `/api/auth/logout` | Выход |
| GET | `/api/auth/check` | Проверка сессии |

### Аккаунты (только admin)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/accounts` | Список аккаунтов |
| POST | `/api/accounts` | Создать аккаунт |
| PUT | `/api/accounts/<username>` | Обновить аккаунт |
| DELETE | `/api/accounts/<username>` | Удалить аккаунт |

### Модели

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/models` | Список всех моделей |
| GET | `/api/config` | Текущая конфигурация |
| POST | `/api/config` | Сохранить конфигурацию |

### Управление сервером

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/server/start` | Запустить llama-server |
| POST | `/api/server/stop` | Остановить сервер |
| POST | `/api/server/restart` | Перезапустить сервер |
| GET | `/api/server/status` | Статус сервера |
| GET | `/api/server/logs` | Логи сервера |

### Пул моделей (multi-model)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/pool/status` | Статус всех слотов |
| POST | `/api/pool/start` | Запустить все слоты |
| POST | `/api/pool/stop` | Остановить все слоты |
| POST | `/api/pool/slot/start` | Запустить слот |
| POST | `/api/pool/slot/stop` | Остановить слот |
| POST | `/api/pool/slot/assign` | Назначить модель на слот |

### Чат

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/chat` | Отправить сообщение (с/без стриминга) |

### Файлы

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/files/upload` | Загрузить файл |
| POST | `/api/files/upload-base64` | Загрузить base64 |
| GET | `/api/files/list` | Список файлов |
| GET | `/api/files/<name>` | Скачать файл |
| DELETE | `/api/files/<name>` | Удалить файл |

### Оркестратор (AI-агент)

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/api/orchestrator/run` | Запустить агентный цикл |
| GET | `/api/orchestrator/status` | Статус оркестратора |
| GET | `/api/orchestrator/providers` | Список LLM-провайдеров |
| GET | `/api/orchestrator/skills` | Список навыков |

---

## Примеры запросов

### Вход в веб-интерфейс
```bash
curl -X POST http://95.174.126.186:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -c cookies.txt \
  -d '{"username": "admin", "password": "hermes_secret_api_key_2026"}'
```

### Проверка сессии
```bash
curl -b cookies.txt http://95.174.126.186:5000/api/auth/check
```

### Проверка статуса
```bash
curl -H "X-Orchestrator-API-Key: hermes_secret_api_key_2026" \
  http://95.174.126.186:5000/api/server/status
```

### Запрос к чату
```bash
curl -X POST http://95.174.126.186:5000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Orchestrator-API-Key: hermes_secret_api_key_2026" \
  -d '{
    "messages": [{"role": "user", "content": "Привет!"}],
    "settings": {"temperature": 0.7}
  }'
```

### Запрос к оркестратору (агент с инструментами)
```bash
curl -X POST http://95.174.126.186:5000/api/orchestrator/run \
  -H "Content-Type: application/json" \
  -H "X-Orchestrator-API-Key: hermes_secret_api_key_2026" \
  -d '{
    "prompt": "Найди информацию о проекте PROJ-001",
    "user_id": "api_user",
    "session_id": "default"
  }'
```

### Загрузка файла
```bash
curl -X POST http://95.174.126.186:5000/api/files/upload \
  -H "X-Orchestrator-API-Key: hermes_secret_api_key_2026" \
  -F "file=@document.pdf"
```

### Python пример
```python
import requests

BASE = "http://95.174.126.186:5000"
HEADERS = {"X-Orchestrator-API-Key": "hermes_secret_api_key_2026"}

# Способ 1: API-ключ в заголовке (для программного доступа)
status = requests.get(f"{BASE}/api/server/status", headers=HEADERS).json()

# Способ 2: Сессия через логин (для веб-доступа)
session = requests.Session()
session.post(f"{BASE}/api/auth/login", json={
    "username": "admin",
    "password": "hermes_secret_api_key_2026"
})
status = session.get(f"{BASE}/api/server/status").json()

# Чат
response = session.post(f"{BASE}/api/chat", json={
    "messages": [{"role": "user", "content": "Что такое квантовые вычисления?"}],
    "settings": {"temperature": 0.7, "max_tokens": 512}
}).json()

print(response["content"])
```

### JavaScript пример
```javascript
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-Orchestrator-API-Key': 'hermes_secret_api_key_2026'
    },
    body: JSON.stringify({
        messages: [{ role: 'user', content: 'Привет!' }],
        stream: false
    })
});
const data = await response.json();
console.log(data.response);
```

### SSE стриминг (потоковая генерация)
```javascript
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-Orchestrator-API-Key': 'hermes_secret_api_key_2026'
    },
    body: JSON.stringify({
        messages: [{ role: 'user', content: 'Расскажи историю' }],
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    process.stdout.write(chunk);
}
```

### Управление аккаунтами (Python)
```python
import requests

BASE = "http://95.174.126.186:5000"

# Войти как admin
session = requests.Session()
session.post(f"{BASE}/api/auth/login", json={
    "username": "admin",
    "password": "hermes_secret_api_key_2026"
})

# Создать пользователя
session.post(f"{BASE}/api/accounts", json={
    "username": "ivanov",
    "password": "secret123",
    "display_name": "Иванов И.И.",
    "role": "user"
})

# Список аккаунтов
accounts = session.get(f"{BASE}/api/accounts").json()
for acc in accounts["accounts"]:
    print(f"{acc['username']} ({acc['role']})")

# Изменить роль
session.put(f"{BASE}/api/accounts/ivanov", json={"role": "admin"})

# Удалить
session.delete(f"{BASE}/api/accounts/ivanov")
```

---

## Конфигурация

### Переменные окружения (.env)
```
ORCHESTRATOR_API_KEY=hermes_secret_api_key_2026
ERP_API_KEY=erp_secret_key_2026
FLASK_SECRET_KEY=auto_generated
```

### runner_config.json
```json
{
  "settings": {
    "host": "127.0.0.1",
    "port": 8080,
    "orchestrator_api_key": "hermes_secret_api_key_2026",
    "allowed_origins": []
  }
}
```

### Доступ по WireGuard
При подключении через WireGuard (10.10.0.x) аутентификация не требуется — трафик уже защищён VPN.

---

## Управление паролями и ключами

### Аккаунты пользователей
- **Хранение:** `accounts.json` (автоматически)
- **Управление:** Настройки → Безопасность → Аккаунты
- **Админ по умолчанию:** логин `admin`, пароль = `orchestrator_api_key`
- **Роли:**
  - `admin` — полный доступ (управление аккаунтами, настройки)
  - `user` — чат, модели, файлы
  - `readonly` — только просмотр

### API-ключ (для программного доступа)
- **Заголовок:** `X-Orchestrator-API-Key`
- **Значение:** То же что `orchestrator_api_key`
- **Где:** `runner_config.json` или `.env`

### Клиентские ключи (multi-client)
- **Где:** `clients/*.json` (например `clients/erp.json`)
- **Заголовки:** `X-Client-ID` + `X-Client-API-Key`
- **Управление:** Через интерфейс (Настройки → Безопасность) или API

### Быстрая смена пароля
```bash
# Windows PowerShell
(Get-Content runner_config.json -Raw | ConvertFrom-Json).settings.orchestrator_api_key = "новый_пароль"
# Или编辑 runner_config.json вручную
```

### Сброс всех ключей
1. Остановите Flask
2. Удалите `clients/*.json`
3. Измените `orchestrator_api_key` в `runner_config.json`
4. Запустите Flask
