# Airflow Summit 2024
## Cloud Composer and GKE Batch

## Repo Structure
### Benchmark
This directory contains the necessary scripts and configuration 
files to evaluate the fine-tuned language model's performance. 
It currently implements a perplexity benchmark, commonly used to 
assess the quality of large language models (LLMs). 
Perplexity measures how well the model predicts a sample of text, 
with lower values indicating better performance.

### Convert
This directory contains scripts and configurations for converting data 
formats between different frameworks. Currently, it includes a u
tility for exporting data from Gemma format to Hugging Face datasets,
facilitating smooth integration with the Hugging Face ecosystem for 
further model training or evaluation.

The script is a modified version of: https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma_kerasnlp_to_vertexai.ipynb

### Finetuning
This directory contains the scripts and configurations for fine-tuning 
large language models (LLMs). It provides the necessary tools to 
adjust pre-trained models on specific tasks or datasets, enabling 
the model to specialize further. Currently, it includes a setup to 
fine-tune models using Keras NLP utilities.

The script is a modified version of: https://ai.google.dev/gemma/docs/lora_tuning

### Preprocessing
This directory contains scripts for preparing and cleaning data 
before it's used for fine-tuning large language models (LLMs). 
The preprocessing steps may include tokenization, data normalization, 
and other techniques to transform raw data into a format suitable for 
model training.
