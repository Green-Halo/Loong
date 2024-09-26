import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, BitsAndBytesConfig
from datasets import load_dataset, DownloadConfig, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import math
import re
import os
from itertools import islice

# 设置模型名称
model_name_1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name_2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name_3 = "SweatyCrayfish/llama-3-8b-quantized"
model_name_4 = "/mnt/2T/Codes/Llama-2-7b-chat-hf"

# 配置量化参数
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_1)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token
model = AutoModelForCausalLM.from_pretrained(model_name_1
                                             , trust_remote_code=True
                                             , device_map="auto"
                                             , torch_dtype = torch.float16
                                            #  , quantization_config=quantization_config
                                             )
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

# 如果有 GPU，可将模型移动到 GPU
device = torch.device("cuda")
# model.to(device)

# 设置下载配置，避免发生连接超时
download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 创建一个目录来保存预下载的数据集
datasets_dir = "/mnt/2T/Codes/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 定义一些通用函数

def compute_perplexity(model, dataloader):
    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=True
            )
            loss = outputs.loss
            total_loss += loss.item()
            total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def extract_answer(output_text):
    # 提取答案中的字母（A、B、C、D、E）
    match = re.search(r'\b([A-E])\b', output_text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        return None

def extract_final_answer(output_text):
    # 尝试匹配 '答案' 后的内容
    match = re.search(r'Answer[::\s]*(.*)', output_text)
    if match:
        return match.group(1).strip()
    else:
        # 如果没有匹配到，尝试其他策略
        lines = output_text.strip().split('\n')
        if lines:
            return lines[-1].strip()
        else:
            return output_text.strip()
        