<h1 align="center">Дипломный проект по программе "Машинное обучение и высоконагруженные системы"</h1>

<h2 align="center">Идентификация искусственно сгенерированных текстов: разработка методов для обеспечения информационной достоверности.</h2>

<h2 align="center">Identification of Artificially Generated Texts: Development of Methods for Ensuring Information Authenticity.</h2>


<h3 align="left">>* Аннотация проекта</h3>

В рамках текущего репозитория выполнена практическая часть ВКР по созданию (и развертыванию) сервиса направленного на идентификацию ИИ сгенерированных политических новостей. Здесть представленны ноутбуки по сбору, обработке данных, построению алгоритмов (\notebooks), так же скрипты для сервисной части: cpu_server + gpu_server, эти 2 сервера связанны между собой и создают единую архитектуру для доступа к сервису по API.


<h3 align="left">* Содержание</h3>

В рамках репозитория предоставлена инструкиция по развертыванию сервиса.

**notebooks** В директории находятся ноутбуки с исследованиями, созданием датасета, обучением моделей идентификации LLM сгенерирвоанных новостей. На основе этих ноутбуков написана ВКР о сервисе идентификации.

**cpu_server** Конфиги, скрипты и прочее необходимое для поднятия сервисной части на CPU сервере (Приложение, БД, Брокер).

**cpu_server** Конфиги, скрипты и прочее необходимое для поднятия сервисной части на GPU сервере (ML сервис детекции).


<h3 align="left">* Описание сервиса</h3>

По итогу развертывания сервисов будет доступно API на CPU сервере, на который можно будет отправлять текст с целью оценки вероятности того, что он был сгенерирован AI моделью. Сервис логирует запрос пользователя в БД (PSQL), отправляет его на GPU при помощи Celery (RABBITMQ), возвращает клиенту идентификатор задачи. GPU сервер слушает очередь и обрабатывает запрос. Затем обновляет лог в БД, записывая результат. Клиент по полученному идентификатору может получить результат выполнения запроса.

**/predict** - Endpoint сервиса CPU, на которую отправляется текст для обработки. В ответ отдается task_id, который так же логируетя в БД и отправляется на GPU сервер.

**/status/task_id** - Endpoint, которая возвращает результат отработки модели по task_id, результат вытягивается из БД.

В директориях CPU и GPU серверов находится инструкция по развертыванию сервисов. Ниже инструкция по настройка серверов - общая, после которого нужно продолжить настройку по инстркции из дирекории ноды.

* 1) Создаем пользователя
* 2) Настройка сервера для разработки (Miniconda, Poetry, Docker, Minicube, ect)
* 3) Настройка окружения для работы сервисов
* 4) Поднимаем сервисы (ссылка на инструкции)
* 5) Запрос к сервису


<h3 align="left">* Настройка серверов</h3>

<h4 align="center">1. Создаем пользователя и даём права sudo  [CPU & GPU]</h4>

<h5 align="left">Для CPU & GPU сервера</h5>

<code>adduser admin</code>

<code>usermod -aG sudo admin</code>

<code>su -- admin</code>


<h4 align="center">2. Настройка сервера для разработки (Miniconda, Poetry, Docker, ect) [CPU & GPU]</h4>

<b>2.1 Устанавливаем Git, net-tools, PSQL</b>

<code>sudo apt update && sudo apt install git && sudo apt install postgresql-client -y && sudo apt install net-tools</code>

<b>2.2 Копируем репозиторий с проектом</b>

<code>git clone https://github.com/Dangennadevich/AI_Text_Detector.git</code>

<code>mv AI_Text_Detector/* .</code>

<b>2.3 Скачиваем Miniconda в корень (~/), Запускаем установку</b>

<code>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh</code>

<code>bash Miniconda3-latest-Linux-x86_64.sh</code>

<b>2.4 Добавляем переменную окружения </b>

<code>nano ~/.bashrc</code>

<code>PATH="$HOME/miniconda3/bin:$PATH" </code>

<b>2.5 Применим изменения в оболочке </b>

<code>source ~/.bashrc </code>

<b>2.6 Проверим python и conda  </b>

<code>which python</code>
<code>conda --version</code>

<b>2.7 Установим poetry для управления зависимостями и проверим установку  </b>

<code>sudo apt update  && sudo apt install python3-poetry</code>

<code>poetry config virtualenvs.create false</code>

<code>poetry --version</code>

<b>2.8 Создадим и виртуальное окружение без библиотек  </b>

<code>conda create --no-default-packages -n cpu_env python=3.11</code>

<code>conda activate cpu_env</code>

<b>2.9 [cpu_env] Установим Docker  </b>

<a href="https://docs.docker.com/engine/install/ubuntu/" target="_blank">docs.docker.com</a>  

<code>sudo groupadd docker</code>

<code>sudo usermod -aG docker $USER</code>

<code>newgrp docker</code>

<b>2.10 [cpu_env]  Настроим DNS-серверы для всех контейнеров, добавить в файл </b>

<code>sudo nano /etc/docker/daemon.json</code>

<code>{
  "dns": ["8.8.8.8", "8.8.4.4"]
}</code>

<b>2.11 [cpu_env]  Перезапустите Docker-демон  </b>

<code> sudo systemctl restart docker </code>

<b>2.12 [cpu_env]  Проверим установку docker </b>

<code> docker run hello-world </code>

<b>2.13 [cpu_env]  Установи Minikube  </b>

<code> curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 </code>

<code> sudo install minikube-linux-amd64 /usr/local/bin/minikube </code>

<code> minikube start --driver=docker </code>

<b>2.14 [cpu_env]  Установи Minikube  </b>

<code> curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" </code>

<code> sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl </code>


<h4 align="center">3. Общая настройка окружения для работы сервера  [CPU & GPU]</h4>

<h5 align="left">Для CPU & GPU сервера</h5>

<b>3.1 [train_env] Команды ниже выполняем из директории cpu_server или gpu_server для CPU и GPU сервера соответсвенно </b>

<code>cd *pu_server/ </code> 

<b>3.3 [train_env] Выполнем инициализацию poetry, устанавливаем библиотеки</b>

<code>poetry install --no-root </code> 

<b>3.4 [train_env]  Создаем файл .env с кредами </b>

<code>nano ~/*pu_server/.env</code>

<code>AWS_SECRET_ACCESS_KEY = 
AWS_ACCESS_KEY_ID = 
POSTGRES_PASSWORD =
RABBITMQ_DEFAULT_USER =
RABBITMQ_DEFAULT_PASS =
CPU_SERVER_IP=
</code> 



<h4 align="center">4. Поднимаем сервисы  [CPU | GPU]</h4>

<b>4.1 Инструкция для CPU сервера находится в ~/cpu_server/README.MD </b>

<b>4.2 Инструкция для GPU сервера находится в ~/gpu_server/README.MD </b>



<h4 align="center">5. Запрос к сервису (IP CPU сервера) [CPU]</h4>

<b>Отправляем запрос на сервис. Поулчаем идентификатор, по которому проверяем ответ.</b>

Ожидается ответ в виде числа, которое является (в будущем) оценкой вероятности того, что текст был сгенерирован моделью, пока это заглушка.

<code>curl -X POST "http://IP_CPU_SERVER:8000/predict" -H "Content-Type: application/json" -d '{"text": "Привет как дела?"}'</code> 

![Alt text](postman-post.png)

<code>curl -X GET "http://IP_CPU_SERVER:8000/status/7a1c9d34-8cba-4d76-9f07-d1993e0c0d18"</code> 

![Alt text](postman-get.png)

Запрос доступен в Flower

![Alt text](flower.png)

<h4 align="center">6. Тест нагрузки</h4>

В примере сервис поднят при помощи minicube на одном узле cpu.

![Alt text](Service-test.png)