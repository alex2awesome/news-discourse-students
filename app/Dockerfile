# Use the official Python image as the base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to avoid binary incompatibility issues
RUN pip install numpy==1.25.0

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Force reinstall numpy to ensure binary compatibility
RUN pip install --force-reinstall numpy==1.25.0

# Copy project files
COPY . .

# Expose the port
EXPOSE 8080

# Command to run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "--workers", "1", "--threads", "4", "--worker-class", "gthread", "run:app"]