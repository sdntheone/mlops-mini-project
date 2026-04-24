FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install \
    dagshub==0.3.34 \
    Flask==3.0.3 \
    mlflow==2.15.0 \
    nltk==3.8.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    python-dotenv==1.0.1 \
    scikit-learn==1.5.1 \
    --default-timeout=1000 \
    --retries=10 \
    --prefer-binary

# ✅ Install ALL required NLTK data (important)
RUN python -m nltk.downloader stopwords wordnet punkt omw-1.4

RUN pip install gunicorn

COPY flask_app/ /app/flask_app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app:app"]