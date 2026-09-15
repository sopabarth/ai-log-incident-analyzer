FROM python:3.14-slim

WORKDIR /app

# Install deps first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY alembic/ alembic/
COPY alembic.ini .

EXPOSE 8000

# Apply any pending migrations, then start the server - keeps schema setup
# out of the app's own startup code (see app/db.py).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
