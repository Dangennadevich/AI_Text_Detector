FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install poetry && \
    pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install transformers
    
COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-root 

COPY . /app

# Команды для запуска сервиса
CMD ["poetry", "run", "python", "main.py"]