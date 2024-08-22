import torch
import json
from transformers import GPTJForCausalLM, AutoTokenizer


local_model_path = "./local"
# Load GPT-J model and tokenizer
model = GPTJForCausalLM.from_pretrained(local_model_path)
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
    #input_data = [inp.detach().cpu().numpy().tolist() for inp in input]
    #output_data = output.detach().cpu().numpy().tolist()
    input_data = [inp.cpu().numpy().tolist() for inp in input]
    output_data = output.cpu().numpy().tolist()

    # Store the input and output
    layer_data[layer_name].append({
        "input": input_data,
        "output": output_data
    })


# Register hooks to capture inputs and outputs for each layer
def register_hooks(model):
    for name, layer in model.named_modules():
        # You can filter by layer type if needed, like torch.nn.Linear
        #if isinstance(layer, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Conv1d)):
        layer.register_forward_hook(capture_hook)


# Register the hooks
register_hooks(model)

# Sample input text
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")

# Run the model to capture inputs and outputs
# Run the model to capture inputs and outputs with gradient computation turned off
with torch.no_grad():  # Disable gradient computation
    outputs = model(**inputs)

# Save the captured layer data to a JSON file
with open('gptj_layer_data.json', 'w') as json_file:
    json.dump(layer_data, json_file, indent=4)

print("Layer data saved to gptj_layer_data.json")
