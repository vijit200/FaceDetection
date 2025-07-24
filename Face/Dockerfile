# Use Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev gcc curl netcat && \
    apt-get clean

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy Django project files
COPY . .

# Collect static files (optional, depends on your setup)
RUN python manage.py collectstatic --noinput

# Run migrations (optional, or run with entrypoint script)
# RUN python manage.py migrate

# Start the server (use gunicorn for production)
CMD ["gunicorn", "Face.wsgi:application", "--bind", "0.0.0.0:8000"]
