# Use the official Python 3.11.9 image as the base
FROM python:3.11.9

# Set the working directory within the container
WORKDIR /app

# Copy the Python script into the container
COPY requirements.txt .
COPY fine-tune.py .

# Run the script using the CMD instruction
RUN pip install -r requirements.txt
CMD ["python", "fine-tune.py"]
