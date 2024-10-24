from utils import *

# GSM8K
print("加载 GSM8K 数据集...")
gsm8k = load_dataset("gsm8k", "main", split="test", cache_dir=datasets_dir, download_config=download_config)

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

            input_text = f"Please answer the following math questions and give the answer in the format 'Answer:' at the end.\n\nQuestion:{question}\n\nAnswer:"

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
                eos_token_id=tokenizer.eos_token_id
            )

            output_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            predicted_answer = extract_final_answer(output_text)

            if compare_answers(predicted_answer, reference_answer):
                correct += 1
            total += 1

    accuracy = correct / total * 100
    return accuracy

gsm8k_accuracy = evaluate_gsm8k(model, gsm8k_encoded)
print(f"GSM8K 数据集的准确率：{gsm8k_accuracy:.2f}%")