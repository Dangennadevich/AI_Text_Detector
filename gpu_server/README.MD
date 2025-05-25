<h1 align="center">Проект по распознованию сгенерированного AI моделью текста</h1>

Проект по распознованию AI-сгенерированного текста, основные компоненты:

    * CPU сервер с FastAPI сервером, на котором будет происходить взаимодействие с пользователем.
    
    * GPU сервер с AI моделью, которая проводит скоринг по запросу с CPU сервера.


<h2 align="center">GPU часть проекта</h2>

Запускается на втором сервере архитектуры - GPU. Сервис ожидает сообщение пользователя от CPU сервера (через Celery), выполняет обработку, результат записывает в БД.

Клиент отправляет запрос на CPU сервер (бекенд веб приложения), CPU сервер отправляет запрос на обработку текста в GPU сервер через брокера сообщений rabbitmq и записывает лог в БД PSQL. GPU получает запрос на обработку через rabbitmq, результат кладет в БД обновляя лог оставленный CPU сервисом.
 

Связи сервиса GPU с другими объектами архитектуры: 
RabbitMQ (Celery) >> GPU >> PSQL


<h2 align="center">Настройка сервисов на GPU</h2>

Выполняется после развертывания необходимого ПО на сервере (README.MD, пункт 1-3) и поднятии инфраструктуры на CPU сервере (cpu_server/README.MD)

<h3 align="center">2. Поднимаем сервисы </h3>

<b>2.1 Создаем секреты </b>

<code>kubectl create secret generic app-secrets \
  --from-env-file=.env \
  -n default</code>

<h3 align="left">2.2 fastapi-service</h3>

<b>2.2.1 Создаем контейнер на основе Dockerfile </b>

<code>eval $(minikube docker-env)</code>

<code>docker build -t ai-model-app:latest .</code>

<b>2.2.2 Применяем манифесты </b>

<code>kubectl apply -f k8s/model-fastapi-deployment.yaml</code>

<code>kubectl apply -f k8s/model-fastapi-service.yaml</code>

<b>2.2.3 Проверка pods </b>

<code>kubectl get pods -l app=model-fastapi-app</code>

<b>2.2.4 Проверка сервиса </b>

<code>curl -v http://localhost:8000</code>

<h3 align="left">2.3 Работа с CronJob</h3>

<b>3.1 Проверка сервиса </b>

<code>cd fit_model</code>

<b>3.2 Создаем контейнер </b>

<code>eval $(minikube docker-env)</code>

<code>docker build -t model-trainer:latest .</code>

<b>3.3 Поднимаем манифесты </b>

<code>kubectl apply -f model-storage.yaml </code>

<code>kubectl get pv</code>

Ожидаем Status = BOUND в pv

<code>kubectl get pvc</code>

<code>kubectl apply -f model-training-cronjob.yaml </code>

<b>3.4 Внеочередной тестовый запуск задачи</b>

<code>kubectl delete job test-job 2>/dev/null</code>

<code>kubectl create job --from=cronjob/model-trainer test-job</code>

<code>kubectl logs job/test-job --follow/code>

Model saved to /data/models/logreg_model_20250324_183640.pkl


<h1 align="center">Файлы из директории GPU сервера</h1>

**main.py** - скрипт для запуска сервиса на GPU сервере. В дальнейшем тут будет работать AI модель по распознованию
сгенерированного текста другой AI моделью, на данный момент тут стоит заглушка np.random.rand()

**Dockerfile** Инструкция для развертывания сервиса с "моделью"

**pyproject.toml** Poetry инструкция для настройки виртуального окружения

**/k8s** Директория с конфигами/манифестами для сервисов k8s