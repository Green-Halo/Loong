from utils import *

# HumanEval
print("加载 HumanEval 数据集...")
humaneval = load_dataset("openai_humaneval", cache_dir=datasets_dir, download_config=download_config)
humaneval_dataset = humaneval['test']

# 6. 在 HumanEval 数据集上评估代码生成能力
print("\n=== 在 HumanEval 数据集上评估代码生成能力 ===")

def generate_code(prompt):
    input_text = f"{prompt}\n# 请完成上述函数的实现。"

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

humaneval_accuracy = evaluate_humaneval(model, humaneval_dataset)
print(f"HumanEval 数据集的代码生成准确率：{humaneval_accuracy:.2f}%")