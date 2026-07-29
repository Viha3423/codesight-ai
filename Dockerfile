# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for compiling C++ extensions in vector DBs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit default port
EXPOSE 7860

# Command to launch the Streamlit app on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]