# Base image (lightweight, with Python 3.10)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (adjust to your entrypoint)
CMD ["python", "run.py"]

