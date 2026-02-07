# X-FLASH Cloud API (Windows)

Локальный FastAPI-прокси к Ollama с публичным HTTPS-доступом через Cloudflare Tunnel.

## Архитектура
Internet clients → Cloudflare Tunnel → FastAPI (`localhost:8000`) → Ollama (`localhost:11434`)

## Быстрый старт (Windows 11)

1. Установите Python 3.11+ и Ollama.
2. Создайте виртуальное окружение и установите зависимости:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Скопируйте и настройте `.env`:
   ```powershell
   copy .env.example .env
   ```
4. Запустите API:
   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

## Админ-команды

Создание пользователя:
```powershell
python -m admin_cli create-user --plan free --credits 250
```

Выдача кредитов:
```powershell
python -m admin_cli add-credits --user-id 1 --credits 100
```

Смена плана:
```powershell
python -m admin_cli set-plan --user-id 1 --plan pro
```

Деактивация:
```powershell
python -m admin_cli set-active --user-id 1 --active false
```

## Проверка API

```powershell
curl -X POST http://127.0.0.1:8000/v1/chat `
  -H "Authorization: Bearer <API_KEY>" `
  -H "Content-Type: application/json" `
  -d '{"chat_id":"demo","messages":[{"role":"user","content":"Hello"}],"model":"xflash-llama31","max_tokens":200,"temperature":0.6}'
```

## Cloudflare Tunnel (пример)

1. Установите `cloudflared`.
2. Запустите туннель:
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8000
   ```

## Конфигурация (.env)

См. `.env.example`.
