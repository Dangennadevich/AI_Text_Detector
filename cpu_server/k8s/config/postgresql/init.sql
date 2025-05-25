CREATE DATABASE mlflow_db;
CREATE DATABASE rabbitmq_db;

\connect rabbitmq_db;

CREATE TABLE tasks_detecting_generated_text (
    task_id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    result TEXT
);