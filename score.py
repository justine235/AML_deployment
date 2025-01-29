import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
from accelerate import infer_auto_device_map

def init():
    """
    Initialize the model and tokenizer while avoiding memory overload.
    """
    global model, tokenizer

    # Get model path
    base_model_path = os.getenv("AZUREML_MODEL_DIR", "/var/azureml-app/azureml-models/Qwen2-VL/1")
    model_path = os.path.join(base_model_path, "Qwen2-VL")

    print(f"🔍 Checking model directory: {model_path}")

    if not os.path.exists(model_path):
        raise RuntimeError(f"❌ Model path {model_path} not found!")

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ config.json not found in {model_path}!")

    print(f"✅ config.json found at {config_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use `device_map="auto"` to manage memory safely
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map="auto",  # Ensures it loads on available memory
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        offload_folder="/tmp/model_offload",  # Offload large weights to disk
        offload_state_dict=True,  # Helps avoid memory fragmentation
        trust_remote_code=True
    )

    print("✅ Model and tokenizer successfully loaded with memory optimization!")

def run(raw_data):
    """
    Process an inference request.
    """
    try:
        data = json.loads(raw_data)
        prompt = data.get("prompt", "Hello, how can I assist you?")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response}

    except Exception as e:
        return {"error": str(e)}
