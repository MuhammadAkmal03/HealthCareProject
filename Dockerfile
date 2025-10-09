# Step 1: Use an official, slim Python runtime as a parent image
FROM python:3.10-slim-buster

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the requirements file first for caching
COPY requirements.txt .

# Step 4: Install dependencies with a longer timeout
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Step 5: Copy your model artifacts into the container
# This is a critical step before copying the app code that uses them.
COPY ./ml_models /app/ml_models

# Step 6: Copy your application code into the container
COPY ./app /app/app

# Step 7: Expose the application port
EXPOSE 8000

# Step 8: Define the command to run your application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]