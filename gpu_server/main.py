import numpy as np
import json
import logging
import pika
import psycopg2
from celery import Celery
import os
from dotenv import load_dotenv
from databases import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Конфигурация
RABBITMQ_DEFAULT_USER = os.getenv("RABBITMQ_DEFAULT_USER")
RABBITMQ_DEFAULT_PASS = os.getenv("RABBITMQ_DEFAULT_PASS")

if not RABBITMQ_DEFAULT_USER or not RABBITMQ_DEFAULT_PASS:
    raise ValueError("RABBITMQ_DEFAULT_USER and RABBITMQ_DEFAULT_PASS must be set in the .env file")

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_USER = os.getenv("POSTGRES_USER")
if not POSTGRES_PASSWORD or not POSTGRES_USER:
    raise ValueError("POSTGRES_PASSWORD and POSTGRES_USER must be set in the .env file")

CPU_SERVER_IP = os.getenv("CPU_SERVER_IP")
if not CPU_SERVER_IP:
    raise ValueError("CPU_SERVER_IP must be set in the .env file")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{CPU_SERVER_IP}:5432/rabbitmq_db"
database = Database(DATABASE_URL, min_size=2, max_size=10)

celery_app = Celery(
    'gpu_tasks',
    broker=f'amqp://{RABBITMQ_DEFAULT_USER}:{RABBITMQ_DEFAULT_PASS}@{CPU_SERVER_IP}:5672//',
    backend='rpc://',
    task_default_queue='tasks_pred_gen_txt'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(name='process_task', bind=True)
def process_task(self, task_id, text):
    try:
        logger.info(f"Processing task {task_id}: {text}")
        
        prob = round(np.random.rand(), 3)
        result = f"Processed text: {text}, probability = {prob}"

        logger.info(f"Result: {result}")
        
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE tasks_detecting_generated_text SET status = 'completed', result = %s WHERE task_id = %s",
                (result, task_id)
            )
            conn.commit()
        
        logger.info(f"Task {task_id} processed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        self.retry(exc=e, countdown=60)

if __name__ == '__main__':
    celery_app.worker_main(
        argv=['worker', '--loglevel=info', '-Q', 'tasks_pred_gen_txt']
    )