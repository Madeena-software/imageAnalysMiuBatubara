FROM python:3.11-slim

WORKDIR /app

COPY . /app

EXPOSE 8000

CMD ["python", "run.py", "8000"]
