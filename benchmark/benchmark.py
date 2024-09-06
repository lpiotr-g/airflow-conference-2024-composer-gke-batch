import os
import shutil
from datasets import load_dataset
import transformers
import torch
import numpy as np
from tqdm import tqdm


def load_local_model(model_path):
    """
    Load a locally saved TensorFlow model for causal language modeling.

    Parameters:
    - model_path (str): Path to the local directory of the fine-tuned model.

    Returns:
    - model: The loaded TensorFlow model.
    - tokenizer: The tokenizer associated with the model.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )

    return model, tokenizer


def compute_perplexity(
    model, tokenizer, dataset_name="wikitext", split="test", max_samples=5
):
    """
    Compute perplexity for a given model on a specified dataset split, limited to a certain number of samples.

    Parameters:
    - model: The HuggingFace model to evaluate.
    - tokenizer: The tokenizer associated with the model.
    - dataset_name (str): The name of the dataset to use (default: 'wikitext').
    - split (str): The dataset split to use for evaluation (default: 'test').
    - max_samples (int): Maximum number of samples to evaluate (default: 5).

    Returns:
    - float: The perplexity score.
    """
    # Load dataset
    dataset = load_dataset(dataset_name, "wikitext-2-v1", split=split)

    # Ensure the model is in evaluation mode
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize variables to track perplexity
    total_loss = 0.0
    total_tokens = 0

    # Evaluate only up to max_samples
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
        if idx >= max_samples:
            break

        # Encode input text and convert to PyTorch tensors
        inputs = tokenizer(
            sample["text"], return_tensors="pt"
        )  # use 'pt' for PyTorch tensors
        input_ids = inputs["input_ids"]

        # Move input to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        with torch.no_grad():
            # Compute model output and loss
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss encountered at sample {idx}. Skipping this sample.")
                continue

            total_loss += loss.detach().item() * input_ids.size(
                1
            )  # Detach tensor from graph and convert to a scalar
            total_tokens += input_ids.size(1)

    # Ensure total_tokens is not zero to avoid division by zero
    if total_tokens == 0:
        print("Total tokens is zero, cannot compute perplexity.")
        return float("inf")

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


def copy_best_model_to_champion(best_model_path, champion_directory="/data/champion"):
    """
    Copy the best model to a champion directory.

    Parameters:
    - best_model_path (str): Path to the best performing model.
    - champion_directory (str): Directory where the best model should be copied.
    """
    if not os.path.exists(champion_directory):
        os.makedirs(champion_directory)

    # Copy model files to the champion directory
    for filename in os.listdir(best_model_path):
        source_file = os.path.join(best_model_path, filename)
        destination_file = os.path.join(champion_directory, filename)
        if os.path.isdir(source_file):
            shutil.copytree(source_file, destination_file, dirs_exist_ok=True)
        else:
            shutil.copy2(source_file, destination_file)


# Load the environment variables for file path and output file name.
MODEL1_PATH = os.getenv("MODEL1_PATH")
MODEL2_PATH = os.getenv("MODEL2_PATH")
MODEL3_PATH = os.getenv("MODEL3_PATH")
MODEL4_PATH = os.getenv("MODEL4_PATH")
CHAMPION_PATH = os.getenv("CHAMPION_PATH")
STEPS = int(os.getenv("STEPS"))

if (
    not MODEL1_PATH
    or not MODEL2_PATH
    or not MODEL3_PATH
    or not MODEL4_PATH
    or not CHAMPION_PATH
    or STEPS <= 0
):
    raise ValueError(
        "MODEL paths, CHAMPION path environment variables must be set and STEPS needs to be > 0"
    )

# List of paths to the fine-tuned models
models = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, MODEL4_PATH]

# Dictionary to store model perplexity
model_perplexities = {}

# Run benchmark and print results
for model_path in models:
    # Copy LLMs to ephemeral storage to speed up IO
    local_model_path = os.path.join("/tmp", model_path[1:])
    shutil.copytree(model_path, local_model_path, dirs_exist_ok=True)
    model, tokenizer = load_local_model(local_model_path)
    perplexity = compute_perplexity(model, tokenizer, max_samples=STEPS)
    model_perplexities[model_path] = perplexity
    print(f"Perplexity for model at {model_path}: {perplexity:.2f}")

# Find the model with the lowest perplexity
best_model_path = min(model_perplexities, key=model_perplexities.get)
print(
    f"Best model found at {best_model_path} with perplexity {model_perplexities[best_model_path]:.2f}"
)

# Copy the best model to the champion directory
copy_best_model_to_champion(best_model_path)
print(f"Best model copied to /data/champion")
