import json
import os

import keras
import keras_nlp

# Ensure that the Keras backend is set correctly before importing Keras models.
os.environ["KERAS_BACKEND"] = "jax"

# Load the environment variables for file path and output file name.
INPUT_FILE = os.getenv("INPUT_FILE")
STEPS = int(os.getenv("STEPS"))
OUTPUT_DIR = os.getenv("OUTPUT_DIRECTORY")

if not INPUT_FILE or not OUTPUT_DIR or STEPS <= 0:
    raise ValueError(
        "INPUT_FILE, OUTPUT_DIR environment variables must be set and STEPS needs to be > 0"
    )


def load_and_format_data(file_path):
    """Loads data from a JSONL file and formats it using a template.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of formatted data strings.
    """
    output_data = []
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
    with open(file_path, "r") as f:
        for line in f:
            loaded_line = json.loads(line)
            rendered_line = template.format(**loaded_line)
            output_data.append(rendered_line)

    return output_data


# Step 1: Load and preprocess data
data = load_and_format_data(INPUT_FILE)
data = data[:STEPS]

# Step 2: Load and configure the model
# Uncomment if you don't have the model downloaded
# gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
# gemma_lm.summary()
# Load the model from local storage
gemma_lm = keras.models.load_model(
    "/data/gemma_2b_en_local.keras", custom_objects={"keras_nlp": keras_nlp}
)
gemma_lm.summary()

# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

# Limit the input sequence length to 256 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 256

# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

# Step 3: Compile and train the model
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=1, batch_size=1)

# Step 4: Store the model
os.makedirs(OUTPUT_DIR, exist_ok=True)
gemma_lm.save_weights(f"{OUTPUT_DIR}/model.weights.h5")
gemma_lm.preprocessor.tokenizer.save_assets(OUTPUT_DIR)
