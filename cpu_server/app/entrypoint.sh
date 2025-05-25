#!/bin/sh
set -e

if [ "$APP_MODE" = "celery_worker" ]; then
  echo "Запуск Celery Worker..."
  exec poetry run celery -A celery_app worker -Q tasks_pred_gen_txt --loglevel=info
elif [ "$APP_MODE" = "flower" ]; then
  echo "Запуск Flower UI..."
  exec poetry run celery -A celery_app flower --port=5555 --broker="${CELERY_BROKER_URL}"
else
  echo "Запуск FastAPI сервиса..."
  exec poetry run gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
fi