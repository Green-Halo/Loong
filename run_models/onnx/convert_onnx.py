from onnxruntime_genai.models.builder import create_model

# Create ONNX model
# model_name = "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-8bit-glue_base"
model_name = None
input_folder = "/home/loong/桌面/Codes/Green_Halo/models/Llama-3.1-8B-Instruct-GPTQ-4bit"
output_folder = "/home/loong/桌面/Codes/Green_Halo/models/Llama-3.1-8B-Instruct-GPTQ-4bit-ONNX"
precision = "int4"
execution_provider = "cuda"
cache_dir = "/home/loong/桌面/Codes/Green_Halo/models/cache"
# cache_dir = os.path.join(".", "cache_dir")

extra_options = {
    "use_qdq": True,
    "use_8bits_moe": False,
    # "enable_cuda_graph": True if execution_provider == "cuda" else False,
    "tunable_op_enable": True if execution_provider == "rocm" else False,
    "tunable_op_tuning_enable": True if execution_provider == "rocm" else False,
}

create_model(model_name, input_folder, output_folder, precision, execution_provider, cache_dir, **extra_options)