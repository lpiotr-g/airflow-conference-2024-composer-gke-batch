# Use the official Python 3.11.9-alpine image as the base
FROM python:3.11.9

# Set the working directory within the container
WORKDIR /app

# Copy the Python scripts into the container
COPY requirements.txt .
COPY export_gemma_to_hf.py .

# Run the script using the CMD instruction
RUN pip install -r requirements.txt

CMD python export_gemma_to_hf.py --weights_file "$MODEL_WEIGHT_FILE" --size 2b --vocab_path "$MODEL_VOCAB_FILE" --output_dir "$OUTPUT_DIR"
