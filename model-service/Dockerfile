# Use Python 3.12 full image (includes build tools)
FROM python:3.12

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements-simple.txt ./

# Install Python dependencies (skip system dependencies for now)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-simple.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/results

# Expose port
EXPOSE 8000

# Health check (simplified without curl)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "main-docker:app", "--host", "0.0.0.0", "--port", "8000"]