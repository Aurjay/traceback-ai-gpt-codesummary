# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for hnswlib and C++11 support
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file into the container
COPY requirements.txt .

# Install the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Copy the .env file into the container
COPY .env .

# Expose the port that the Flask app runs on
EXPOSE 8080

# Set the entrypoint command to run the Flask app
CMD ["python", "api/new-gpt-codesummary/main.py"]
