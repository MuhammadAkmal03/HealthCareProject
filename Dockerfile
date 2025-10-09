# Step 1: Use an official, slim Python runtime as a parent image
FROM python:3.10-slim-buster

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Step 5: Copy your application code and model artifacts
COPY ./app /app/app
COPY ./ml_models /app/ml_models

# Step 6: Expose the application port
EXPOSE 8000

# Step 7: Define the command to run your application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
