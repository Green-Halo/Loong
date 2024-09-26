import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, BitsAndBytesConfig
from datasets import load_dataset, DownloadConfig, get_dataset_config_names, concatenate_datasets
from tqdm import tqdm
import math
import re
import os

import logging
from transformers import logging as transformers_logging

# 设置日志记录级别为 ERROR
logging.basicConfig(level=logging.ERROR)
transformers_logging.set_verbosity_error()

# 设置模型名称
model_name_1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name_2 = "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-8bit"
model_name_3 = "GreenHalo/Llama-3.1-8B-Instruct-GPTQ-4bit"

model_name = model_name_2

# 配置超参数
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
batch_size = 4

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token
model = AutoModelForCausalLM.from_pretrained(model_name
                                             , trust_remote_code=True
                                             , device_map="auto"
                                             , torch_dtype = torch.float16
                                            #  , quantization_config=quantization_config
                                             )
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

# 如果有 GPU，可将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 设置下载配置，避免发生连接超时
download_config = DownloadConfig()
download_config.num_proc = 8  # 根据您的网络情况调整
download_config.max_retries = 5

# 创建一个目录来保存预下载的数据集
datasets_dir = "/mnt/2T/Codes/Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# 预先加载所有数据集
print("开始预载所有数据集...")

# Wikitext-2
print("加载 Wikitext-2 数据集...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=datasets_dir, download_config=download_config)

# PTB (Penn Treebank)
print("加载 PTB (Penn Treebank) 数据集...")
ptb = load_dataset("datasets/ptb.py", "penn_treebank", split="test", cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True)

# CommonSenseQA
print("加载 CommonSenseQA 数据集...")
commonsense_qa = load_dataset("commonsense_qa", split="validation", cache_dir=datasets_dir, download_config=download_config)

# MMLU
print("加载 MMLU 数据集...")
mmlu = load_dataset("cais/mmlu", "all", split="test", cache_dir=datasets_dir, download_config=download_config)

# GSM8K
print("加载 GSM8K 数据集...")
gsm8k = load_dataset("gsm8k", "main", split="test", cache_dir=datasets_dir, download_config=download_config)

# HumanEval
print("加载 HumanEval 数据集...")
humaneval = load_dataset("openai_humaneval", split="test", cache_dir=datasets_dir, download_config=download_config)

# C4
# print("加载 C4：Colossal Clean Crawled Corpus数据集...")
# c4 = load_dataset("datasets/c4.py", "en", split="validation", streaming=True, cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True)

print("所有数据集已加载完成。")

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

# 1. 在 Wikitext-2 上计算困惑度
print("\n=== 在 Wikitext-2 数据集上计算困惑度 ===")

def preprocess_wikitext(batch):
    inputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    inputs['labels'] = inputs['input_ids']
    return inputs

wikitext_encoded = wikitext.map(preprocess_wikitext, batched=True, remove_columns=['text'])
wikitext_dataloader = DataLoader(wikitext_encoded, batch_size=batch_size, collate_fn=default_data_collator)

wikitext_perplexity = compute_perplexity(model, wikitext_dataloader)
print(f"Wikitext-2 数据集的困惑度：{wikitext_perplexity}")

# 2. 在 PTB 数据集上计算困惑度
print("\n=== 在 PTB 数据集上计算困惑度 ===")

def preprocess_ptb(batch):
    inputs = tokenizer(
        batch['sentence'],
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    inputs['labels'] = inputs['input_ids']
    return inputs

ptb_encoded = ptb.map(preprocess_ptb, batched=True, remove_columns=['sentence'])
ptb_dataloader = DataLoader(ptb_encoded, batch_size=batch_size, collate_fn=default_data_collator)

ptb_perplexity = compute_perplexity(model, ptb_dataloader)
print(f"PTB 数据集的困惑度：{ptb_perplexity}")

# 3. 在 CommonSenseQA 数据集上评估准确率
print("\n=== 在 CommonSenseQA 数据集上评估准确率 ===")

def format_commonsenseqa(example):
    question = example["question"]
    choices = example["choices"]["text"]
    answer_key = example["answerKey"]
    answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    label = answer_mapping[answer_key]

    input_text = f"Please answer the following questions and select the correct option. \n\nQuestion: {question}\n\nOptions:\n"
    options = ['A', 'B', 'C', 'D', 'E']
    for i, choice in enumerate(choices):
        input_text += f"{options[i]}. {choice}\n"
    input_text += "\nAnswer: "

    return {"input_text": input_text, "label": label}

commonsense_qa_encoded = commonsense_qa.map(format_commonsenseqa)

def evaluate_commonsenseqa(model, dataset):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for example in tqdm(dataset):
            input_text = example["input_text"]
            label = example["label"]

            # 使用 tokenizer(...) 获取 attention_mask
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 10,
                num_return_sequences=1,
                do_sample=False
            )

            output_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            predicted_answer = extract_answer(output_text)
            if predicted_answer is None:
                predicted_index = -1
            else:
                answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
                predicted_index = answer_mapping.get(predicted_answer, -1)

            if predicted_index == label:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    return accuracy

commonsense_qa_accuracy = evaluate_commonsenseqa(model, commonsense_qa_encoded)
print(f"CommonSenseQA 数据集的准确率：{commonsense_qa_accuracy:.2f}%")

# 4. 在 MMLU 数据集上评估准确率
print("\n=== 在 MMLU 数据集上评估准确率 ===")

def format_mmlu(example):
    question = example["question"]
    choices = example["choices"]
    label = example["answer"]  # 直接使用整数标签

    input_text = f"Please choose the correct answer based on the following questions. \n\nQuestion: {question}\n\nOptions:\n"
    options = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        input_text += f"{options[i]}. {choice}\n"
    input_text += "\nAnswer："

    return {"input_text": input_text, "label": label}


mmlu_encoded = mmlu.map(format_mmlu)

def evaluate_mmlu(model, dataset):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for example in tqdm(dataset):
            input_text = example["input_text"]
            label = example["label"]  # 整数标签

            # 使用 tokenizer(...) 获取 attention_mask
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 10,
                num_return_sequences=1,
                do_sample=False
            )

            output_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            predicted_answer = extract_answer(output_text)
            if predicted_answer is None:
                predicted_index = -1
            else:
                answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                predicted_index = answer_mapping.get(predicted_answer.upper(), -1)

            if predicted_index == label:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    return accuracy

mmlu_accuracy = evaluate_mmlu(model, mmlu_encoded)
print(f"MMLU 数据集的准确率：{mmlu_accuracy:.2f}%")

# 5. 在 GSM8K 数据集上评估准确率
print("\n=== 在 GSM8K 数据集上评估准确率 ===")

def format_gsm8k(example):
    question = example["question"]
    answer = example["answer"]

    # 对答案进行清理，提取最终的数字答案
    match = re.search(r'####\s*(.*)', answer)
    if match:
        cleaned_answer = match.group(1).strip()
    else:
        cleaned_answer = answer.strip()

    return {"question": question, "answer": cleaned_answer}

gsm8k_encoded = gsm8k.map(format_gsm8k)

def compare_answers(predicted, reference):
    try:
        pred_value = float(predicted)
        ref_value = float(reference)
        return abs(pred_value - ref_value) < 1e-1
    except ValueError:
        return predicted.strip() == reference.strip()

def evaluate_gsm8k(model, dataset):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for example in tqdm(dataset):
            question = example["question"]
            reference_answer = example["answer"]

            input_text = f"Please answer the following math questions and give the answer in the format 'Answer:' at the end. \n\nQuestion:{question}\n\nAnswer:"

            inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

            output = model.generate(
                inputs,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                # temperature=0.7,
                # top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )

            output_text = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

            predicted_answer = extract_final_answer(output_text)

            if compare_answers(predicted_answer, reference_answer):
                correct += 1
            total += 1

    accuracy = correct / total * 100
    return accuracy


gsm8k_accuracy = evaluate_gsm8k(model, gsm8k_encoded)
print(f"GSM8K 数据集的准确率：{gsm8k_accuracy:.2f}%")

# 6. 在 HumanEval 数据集上评估代码生成能力
print("\n=== 在 HumanEval 数据集上评估代码生成能力 ===")

def generate_code(prompt):
    input_text = f"{prompt}\n# Please complete the implementation of the above function."

    # 使用 tokenizer(...) 获取 attention_mask
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        num_beams=1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_code = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated_code

def evaluate_code(task_id, prompt, test_code, generated_code):
    full_code = prompt + generated_code + "\n" + test_code

    namespace = {}

    try:
        exec(full_code, namespace)
        return True
    except Exception:
        return False

def evaluate_humaneval(model, dataset):
    correct = 0
    total = 0

    for example in tqdm(dataset):
        task_id = example['task_id']
        prompt = example['prompt']
        test_code = example['test']
        canonical_solution = example['canonical_solution']

        generated_code = generate_code(prompt)

        success = evaluate_code(task_id, prompt, test_code, generated_code)

        if success:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    return accuracy

humaneval_accuracy = evaluate_humaneval(model, humaneval)
print(f"HumanEval 数据集的代码生成准确率：{humaneval_accuracy:.2f}%")

# 7. 在 C4 数据集上计算困惑度
# print("\n=== 在 C4数据集上计算困惑度 ===")

# def preprocess_c4(batch):
#     inputs = tokenizer(
#         batch['text'],
#         truncation=True,
#         max_length=512,
#         return_tensors='pt',
#         padding='max_length'
#     )
#     inputs['labels'] = inputs['input_ids']
#     return inputs

# c4_encoded = c4.map(preprocess_c4, batched=True)
# c4_dataloader = DataLoader(c4_encoded, batch_size=batch_size)

# c4_perplexity = compute_perplexity(model, c4_dataloader)
# print(f"C4 数据集的困惑度：{c4_perplexity}")

print("\n=== 评估完成 ===")
print(f"Wikitext-2 困惑度：{wikitext_perplexity}")
print(f"PTB 困惑度：{ptb_perplexity}")
print(f"CommonSenseQA 准确率：{commonsense_qa_accuracy:.2f}%")
print(f"MMLU 准确率：{mmlu_accuracy:.2f}%")
print(f"GSM8K 准确率：{gsm8k_accuracy:.2f}%")
print(f"HumanEval 代码生成准确率：{humaneval_accuracy:.2f}%")
# print(f"C4 困惑度：{c4_perplexity}")