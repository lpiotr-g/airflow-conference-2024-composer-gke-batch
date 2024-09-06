import json
import random
import os
import sys

input_file = os.getenv("INPUT_FILE")
output_file = os.getenv("OUTPUT_FILE")
num_items = os.getenv("NUM_ITEMS")

# Check for missing or empty environment variables
if input_file is None or output_file is None or num_items is None:
    print(
        "Error: The following environment variables are required: INPUT_FILE, OUTPUT_FILE, NUM_ITEMS"
    )
    sys.exit(1)  # Exit with an error code

num_items = int(num_items)  # Convert NUM_ITEMS to an integer

data_without_context = []
with open(input_file, "r") as f:
    for line in f:
        features = json.loads(line)
        if not features["context"]:
            data_without_context.append(
                {
                    "instruction": features["instruction"],
                    "response": features["response"],
                }
            )

if len(data_without_context) < num_items:
    print(
        f"Warning: There are only {len(data_without_context)} items without context in the file."
    )

sampled_data = random.sample(
    data_without_context, min(num_items, len(data_without_context))
)

with open(output_file, "w") as f:
    for item in sampled_data:
        json.dump(item, f)
        f.write("\n")

print(
    f"Sampled {len(sampled_data)} items without context and saved to {output_file} in JSONL format"
)
