from celery import Celery
import os

RABBITMQ_USER = os.getenv("RABBITMQ_DEFAULT_USER")
RABBITMQ_PASS = os.getenv("RABBITMQ_DEFAULT_PASS")

celery_app = Celery(
    'tasks',
    broker=f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@rabbitmq:5672//',
    backend='rpc://',
    include=['main'],
    task_default_exchange='celery',
    task_default_exchange_type='direct',
    task_default_queue='tasks_pred_gen_txt',
    task_default_routing_key='tasks_pred_gen_txt',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True
)


celery_app.conf.task_routes = {
    'process_task': {
        'queue': 'tasks_pred_gen_txt',
        'exchange': 'celery',
        'routing_key': 'tasks_pred_gen_txt'
    }
}