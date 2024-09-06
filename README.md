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

## Useful Resources
### GKE Batch and Kueue
- https://github.com/GoogleCloudPlatform/ai-on-gke/tree/main/best-practices/gke-batch-refarch
- https://cloud.google.com/blog/products/containers-kubernetes/serving-gemma-on-google-kubernetes-engine-deep-dive
- https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e
- https://kueue.sigs.k8s.io/docs/overview/
- https://kubernetes.io/docs/concepts/workloads/controllers/job/
- https://www.youtube.com/watch?v=HWTNCTaKZ_o
- https://www.youtube.com/watch?v=YwSZUdU3iRY

### Cloud Composer and GKE Batch/Vertex Operators
- https://cloud.google.com/composer/docs/composer-2/use-gke-operator
- https://airflow.apache.org/docs/apache-airflow-providers-google/stable/operators/cloud/kubernetes_engine.html
- https://airflow.apache.org/docs/apache-airflow-providers-google/stable/operators/cloud/vertex_ai.html
- https://cloud.google.com/blog/products/data-analytics/announcing-apache-airflow-operators-for-google-generative-ai

### LLMs
- https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/
- https://ai.google.dev/gemma/docs/lora_tuning
- https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
- https://www.youtube.com/watch?v=zjkBMFhNj_g
- https://www.youtube.com/watch?v=l8pRSuU81PU
