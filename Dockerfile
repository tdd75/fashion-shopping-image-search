FROM python:3.8

WORKDIR /app/image_search

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .