FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-root 

COPY . /app

# Команды для запуска сервиса
CMD ["poetry", "run", "python", "main.py"]