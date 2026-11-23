from celery import Celery
from app.core.config import settings


app = Celery('plagiarism_detection')

# Configure Celery with explicit settings
app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Crash safety: ack only after the task finishes, so a worker dying
    # mid-batch redelivers the job instead of silently losing it.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)



# Import tasks
# autodiscover_tasks only finds app/services/tasks.py; our tasks live in
# batch_processing.py, so import it explicitly for worker registration.
app.autodiscover_tasks(['app.services'])
app.conf.imports = ('app.services.batch_processing',)