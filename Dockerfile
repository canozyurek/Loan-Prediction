FROM python:3.7
COPY . .
COPY /helpers /helpers
RUN pip install --no-cache-dir -r  requirements.txt
CMD ["gunicorn", "--preload", "main:app", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080"]