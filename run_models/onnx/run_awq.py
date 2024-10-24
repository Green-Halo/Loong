import argparse
import os

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from onnxruntime_genai.models.builder import create_model
import onnxruntime_genai as og

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        # required=True,
        help="Folder to load PyTorch model and associated files from",
    )

    parser.add_argument(
        "-q",
        "--quant_path",
        # required=True,
        help="Folder to save AWQ-quantized PyTorch model and associated files in",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        # required=True,
        help="Folder to save AWQ-quantized ONNX model and associated files in",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        default="cuda",
        help="Target execution provider to apply quantization (e.g. dml, cuda)",
    )

    parser.add_argument(
        "--use_qdq",
        action="store_true",
        help="Use the QDQ decomposition for quantized MatMul instead of the MatMulNBits operator",
    )

    args = parser.parse_args()
    return args

def run_model(args):
    model = og.Model(args.output_path)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()

    # Override any default search options in `genai_config.json`
    search_options = {
        'min_length': 1,
        'max_length': 2048,
    }

    # Chat template for Phi-3 (replace with the chat template for your model)
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
    while True:
        text = input("Input: ")
        if not text:
            print("Error, input cannot be empty")
            continue
        prompt = f'{chat_template.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)

        print()
        print("Output: ", end='', flush=True)

        try:
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()

                new_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()

        # Delete the generator to free the captured graph for the next generator, if graph capture is enabled
        del generator

def main():
    # args = parse_args()

    # Create ONNX model
    model_name = "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base"
    input_folder = "/home/loong/桌面/Codes/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base"
    output_folder = "/home/loong/桌面/Codes/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX"
    precision = "int4"
    execution_provider = "cuda"
    cache_dir = os.path.join(".", "cache_dir")

    extra_options = {
        "use_qdq": True,
    }

    create_model(model_name, 
                 input_folder, 
                 output_folder, precision, execution_provider, cache_dir, **extra_options)

    # Run ONNX model

if __name__ == "__main__":
    # main()
    args = parse_args()
    args.output_path = "/home/loong/桌面/Codes/Green_Halo/models/Llama-3.1-8B-Instruct-GPTQ-4bit-glue_base-ONNX"
    run_model(args)
