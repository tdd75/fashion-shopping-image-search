FROM python:3.8

WORKDIR /app/image_search

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY . .