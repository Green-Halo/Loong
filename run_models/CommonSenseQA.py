from utils import *

# CommonSenseQA
print("加载 CommonSenseQA 数据集...")
commonsense_qa = load_dataset("commonsense_qa", cache_dir=datasets_dir, download_config=download_config)

# 3. 在 CommonSenseQA 数据集上评估准确率
print("\n=== 在 CommonSenseQA 数据集上评估准确率 ===")

commonsense_qa_validation = commonsense_qa['validation']

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

commonsense_qa_encoded = commonsense_qa_validation.map(format_commonsenseqa)

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