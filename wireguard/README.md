# WireGuard VPN для Llama Runner MIMO

## Как это работает

```
Клиент (peer)                          Сервер (ваш ПК)
┌─────────────────┐                    ┌─────────────────────────┐
│  WireGuard VPN   │◄──────────────────►│  WireGuard (wg0)        │
│  10.10.0.2       │   зашифрованный    │  10.10.0.1              │
│                  │   туннель           │                         │
│  Доступ:         │                    │  Flask: 127.0.0.1:5000  │
│  http://10.10.0.1:5000               │  llama: 127.0.0.1:8080  │
└─────────────────┘                    └─────────────────────────┘
```

- Пир подключается к VPN, получает IP 10.10.0.2
- Доступ к Flask: `http://10.10.0.1:5000`
- Доступ к llama-server: `http://10.10.0.1:8080`
- Весь трафик зашифрован, никаких API-ключей по HTTPS не нужно

## Установка

1. Скачать WireGuard: https://www.wireguard.com/install/
2. Скопировать `server.conf` в `C:\Program Files\WireGuard\Data\Configurations\`
3. Запустить WireGuard GUI → импортировать `server.conf`
4. Нажать "Activate"

Для клиента — скопировать `client-peer.conf` и импортировать в его WireGuard.

## Генерация ключей

На сервере:
```bash
wg genkey | tee server_private.key | wg pubkey > server_public.key
```

Для каждого клиента:
```bash
wg genkey | tee client_private.key | wg pubkey > client_public.key
```
