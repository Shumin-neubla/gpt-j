import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a smaller, faster model
local_model_path = "./distilgpt2"  # Using the smaller 'distilgpt2' model
device = "cpu"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Dictionary to store inputs and outputs for each layer
layer_data = {}

# Hook function to capture input and output
def capture_hook(module, input, output):
    layer_name = module.__class__.__name__

    # If the layer is already in the dictionary, append to the list
    if layer_name not in layer_data:
        layer_data[layer_name] = []

    # Convert tensors to Python lists for easy serialization
    input_data = [inp.cpu().numpy().tolist() for inp in input]
    output_data = output  # Handle other types of outputs if necessary

    print(f"layer name is  {layer_name}")
    # Store the input and output
    layer_data[layer_name].append({
        "input": input_data,
        "output": output_data
    })

# Register hooks to capture inputs and outputs for each layer
def register_hooks(model):
    for name, layer in model.named_modules():
        layer.register_forward_hook(capture_hook)

# Register the hooks
register_hooks(model)

# Sample input text
prompt = "In a shocking finding"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Run the model to capture inputs and outputs with gradient computation turned off
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,              # Enable sampling to introduce randomness
        temperature=0.9,
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )

gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Convert tensors to lists
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [tensor_to_list(item) for item in obj]
    # Check if obj is JSON serializable
    try:
        json.dumps(obj)
    except Exception as e:
        print(f"Object of type {type(obj)} is not JSON serializable")
        print(f"Exception occurred: {type(e).__name__} - {e}")
    finally:
        return obj

# Convert the dictionary, ensuring all tensors are converted
layer_data_serializable = tensor_to_list(layer_data)

with open('distilgpt2_layer_data.json', 'w') as json_file:
    json.dump(layer_data_serializable, json_file, indent=4)

print("Layer data saved to distilgpt2_layer_data.json")
