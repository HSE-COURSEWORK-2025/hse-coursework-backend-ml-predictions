# HSE Coursework: ML Predictions

Этот репозиторий содержит процесс для генерации предсказаний состояния здоровья пользователя на основе агрегированных данных с помощью ML-моделей.

## Основные возможности
- Генерация предсказаний по нескольким диагнозам (бессонница/апноэ, гипертония, депрессия)
- Использование обученных ML-моделей (pickle-файлы)
- Отправка уведомлений о ходе анализа
- Хранение результатов предсказаний в базе данных


## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/your-username/hse-coursework-backend-ml-predictions.git
cd hse-coursework-backend-ml-predictions
```

### 2. Сборка Docker-образа
```bash
docker build -f dockerfile.dag -t predict_using_ml:latest .
```

### 3. Запуск контейнера
```bash
docker run --env-file .env.prod predict_using_ml:latest --email <user@example.com>
```
Где `<user@example.com>` — email пользователя, для которого требуется сгенерировать предсказания.

### 4. Переменные окружения
Используйте `.env.dev` для разработки и `.env.prod` для продакшена. Примеры переменных:
```
DATA_COLLECTION_API_BASE_URL=http://localhost:8082
AUTH_API_BASE_URL=http://localhost:8081
REDIS_HOST=localhost
NOTIFICATIONS_API_BASE_URL=http://localhost:8083/notifications-api/api/v1/notifications
```

### 5. Развёртывание в Kubernetes
Скрипт для развертывания:
```bash
./deploy.sh
```

## Структура проекта
- `run.py` — основной скрипт запуска ML-предсказаний
- `make_predictions_funcs.py` — функции для подготовки данных и вызова моделей
- `ml_models/` — директория с кодом для работы с ML-моделями
- `ml_models_files/` — директория с pickle-файлами обученных моделей (игнорируется в git)
- `models.py` — Pydantic-модели для валидации входных и выходных данных
- `settings.py` — конфигурация приложения
- `notifications.py` — отправка email-уведомлений
- `redis.py` — клиент для Redis
- `requirements.txt` — зависимости Python
- `dockerfile.dag` — Dockerfile для сборки образа

## Пример запуска
```bash
docker run --env-file .env.prod predict_using_ml:latest --email user@example.com
```

## Пример входных данных

В качестве входных данных используется email пользователя, зарегистрированного в системе. Все необходимые данные (сон, активность, пульс и др.) агрегируются автоматически из баз данных.

---

Для корректной работы требуется наличие обученных моделей в директории `ml_models_files/` (файлы с расширением `.pkl`).
Они могут быть получены путем запуска Jupyter-ноубуков из [репозитория с моделями ML](https://github.com/HSE-COURSEWORK-2025/hse-coursework-backend-ml-predictions) 