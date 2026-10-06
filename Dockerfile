FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure data directory has correct permissions
RUN chmod -R 777 /app/data

# Expose port (Render uses standard web ports, Hugging Face Spaces uses 7860)
# Defaulting to 10000 for Render
ENV PORT=10000
EXPOSE 10000

# Run with Gunicorn
CMD gunicorn -b 0.0.0.0:$PORT app:app --timeout 120 --workers 1 --threads 4
