# Use a stable Python image
FROM python:3.11-slim

# Avoid Python writing .pyc files & make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set app directory
WORKDIR /app

# Install system deps (needed for numpy/pandas/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including model + scalers + templates)
COPY . .

# Expose port (Railway / Docker will use this)
ENV PORT=8080
EXPOSE 8080

# Gunicorn start command
# 'app:app' = app.py (module) and app = Flask instance
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
