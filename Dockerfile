
FROM --platform=linux/amd64 python:3.10-slim-bullseye


ENV PYTHONUNBUFFERED=1
ENV LANG C.UTF-8


RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

CMD ["python", "-m", "src.main"]
