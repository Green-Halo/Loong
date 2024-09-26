from utils import *

# MMLU
print("加载 MMLU 数据集...")
mmlu = load_dataset("cais/mmlu", "all", split="test", cache_dir=datasets_dir, download_config=download_config)

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