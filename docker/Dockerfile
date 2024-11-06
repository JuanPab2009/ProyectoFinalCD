# Use a base Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (if necessary for your application)
EXPOSE 8000

# Set the entry point or command to run your application
CMD ["python", "Training_pipeline.py"]

