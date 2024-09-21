from celery import Celery

# Initialize the Celery instance
celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

import main  # Explicitly import main to register tasks

celery.autodiscover_tasks(["main"])
