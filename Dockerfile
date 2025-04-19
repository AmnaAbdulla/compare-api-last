FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "main:main_app"]