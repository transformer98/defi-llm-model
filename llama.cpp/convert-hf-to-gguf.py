import os
import sys
import subprocess

def convert_model_to_gguf(model_dir, out_file=None, outtype="f16"):
    if not os.path.isdir(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        sys.exit(1)

    conversion_script = "convert-hf-to-gguf.py"

    if not os.path.exists(conversion_script):
        print("❌ Make sure you're inside llama.cpp and have convert-hf-to-gguf.py available.")
        sys.exit(1)

    print(f"🔁 Converting {model_dir} to GGUF format...")

    cmd = [
        "python3",
        conversion_script,
        model_dir,
        "--outtype", outtype
    ]
    if out_file:
        cmd += ["--outfile", out_file]

    subprocess.run(cmd)

def quantize_model(input_gguf, output_gguf, quant_type="Q4_0"):
    print(f"⚡ Quantizing {input_gguf} → {output_gguf} using {quant_type}...")
    subprocess.run(["./quantize", input_gguf, output_gguf, quant_type])

if __name__ == "__main__":
    MODEL_DIR = "../my-swap-model"
    OUTFILE = "../my-swap-model.gguf"
    QUANTIZED_FILE = "../my-swap-model-q4.gguf"

    convert_model_to_gguf(MODEL_DIR, out_file=OUTFILE, outtype="f16")
    quantize_model(OUTFILE, QUANTIZED_FILE, quant_type="Q4_0")

    print("✅ Conversion and quantization complete.")
