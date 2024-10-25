import onnxruntime_genai as og
import numpy as np
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import time

class LLAMA:
    def __init__(self, model_path, max_length):
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.max_length = max_length
        self.batch_size_for_cuda_graph = 1
        self.params = og.GeneratorParams(self.model)
        self.search_options = {name: getattr(self, name) for name in ['do_sample', 'max_length', 'min_length', 'top_p', 'top_k', 'temperature', 'repetition_penalty'] if hasattr(self, name)}
        self.params.set_search_options(**self.search_options)
        
    def run(self, chat_template, prompt):
        if chat_template.count('{') != 1 or chat_template.count('}') != 1:
            print("Error, chat template must have exactly one pair of curly braces, e.g., 'Prompt: {input}'")
            exit(1)
        prompt_formatted = chat_template.format(input=prompt)
                
        input_tokens = self.tokenizer.encode(prompt_formatted)

        # Set the batch size for the CUDA graph to the number of prompts if the user didn't specify a batch size
        self.params.try_graph_capture_with_max_batch_size(1)
        self.params.input_ids = input_tokens

        # output_tokens = model.generate(params)

        # for i in range(len(prompts)):
        #     # print(f'Prompt #{i}: {prompts[i]}')
        #     # print()
        #     print(tokenizer.decode(output_tokens[i]))
        #     print()
        
        generator = og.Generator(self.model, self.params)
        output = ""
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output += self.tokenizer.decode(new_token)
        return output.strip()
        # file.write(output+'\n')

glue_tasks = ["sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]
datasets = {}

# 加载所有的 GLUE 任务数据集
for task in glue_tasks:
    if task == "mnli":
        # 加载 validation_matched 和 validation_mismatched，并将它们合并
        dataset_matched = load_dataset("glue", task, split="validation_matched[:1%]")
        dataset_mismatched = load_dataset("glue", task, split="validation_mismatched[:1%]")
        # 使用 concatenate_datasets 方法合并两个数据集
        datasets[task] = concatenate_datasets([dataset_matched, dataset_mismatched])
    else:
        # 其他任务加载 validation 集
        datasets[task] = load_dataset("glue", task, split="validation[:2%]")

datas = defaultdict(list)

def format_example(data_task, example):
    if data_task == "mrpc":
        return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
    elif data_task == "qqp":
        return f"question1: '{example['question1']}' & question2: '{example['question2']}'"
    elif data_task == "mnli":
        return f"premise: '{example['premise']}' & hypothesis: '{example['hypothesis']}'"
    elif data_task == "qnli":
        return f"question: '{example['question']}' & sentence: '{example['sentence']}'"
    elif data_task == "rte":
        return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
    elif data_task == "wnli":
        return f"sentence1: '{example['sentence1']}' & sentence2: '{example['sentence2']}'"
    elif data_task == "sst2":
        return f"{example['sentence']}"
    else:
        return None

# 遍历并处理每个数据集
for data_task, dataset in datasets.items():
    for example in dataset:
        formatted_text = format_example(data_task, example)
        if formatted_text:
            datas[data_task].append({
                'text': formatted_text,
                'label': example['label']
            })

template={}

for key in glue_tasks:
    if key == "sst2":
        template[key] = """
        Prompt: "{input}"
        Instruct: Answer as less as possible. Please determine the sentiment of this above sentence in Prompt. The options are: 0 if the sentence is negative. 1 if the sentence is positive.          No analyses or explanations.Only respond with 0 or 1.
        """
    if key == "mrpc":
        template[key] = """
        Prompt: "{input}"
        Instruct: Answer as less as possible. Please determine whether the two sentences above in Prompt are equivalent, and return 1 if they are, or 0 if they are not.       No analyses or explanations.Only respond with 0 or 1.
        """
    elif key == "qqp":
        template[key] = """
        Prompt: "{input}"
        Instruct: Answer as less as possible. Please determine whether a pair of questions above in Prompt are semantically equivalent, and return 1 if they are, or 0 if they are not.         You can only return 0 or 1.
        """
    elif key == "mnli":
        template[key]="""
        Prompt: "{input}"
        Instruct: Answer as less as possible. From the above premise sentence and hypothesis sentence in Prompt, Please determine the relationship between the two. The options are: 0 if the premise entails the hypothesis. 1 if the relationship is neutral. 2 if the hypothesis contradicts the premise.        Here are your sentences to evaluate: Premise: [Insert Premise Sentence Here] & Hypothesis: [Insert Hypothesis Sentence Here]
    """
    elif key=="qnli":
        template[key] = """
        Prompt: "{input}"
        Instruct: Answer as less as possible. From the above question and sentence in Prompt, Please determine whether the sentence contains the answer to the question. The options are: 0 if the sentence contains the answer. 1 if the sentence does not contains the answer.        Here are your sentences to evaluate: question: [Insert Question Here] & sentence: [Insert Sentence Here]. No analyses or explanations. Only respond with 0, 1, or 2.
        """
    elif key=="rte":
        template[key] ="""
        Prompt: "{input}"
        Instruct: Answer as less as possible. From the above two sentences in Prompt, Please determine whether two sentences are entailments. The options are: 0 if the sentences are entailments. 1 if the sentences are not entailments.           Here are your sentences to evaluate: sentence1: [Insert Sentence Here] & sentence2: [Insert Sentence Here]. No analyses or explanations.Only respond with 0 or 1.
        """
    elif key=="wnli":
        template[key] = """
        Prompt: "{input}"
        Instruct: Answer as less as possible. From the above question and sentence in Prompt, Please determine whether the sentences contain the answer to the question. The options are: 0 if the sentence contains the answer. 1 if the sentence does not contains the answer.    Here are your sentences to evaluate: question: [Insert Question Here] & sentences: [Insert Sentence Here]. No analyses or explanations. Only respond with 0 or 1.
        """

model_name = "Llama-3.1-8B-Instruct-AWQ-4bit-ONNX"
model_path = f"models/{model_name}"

model = LLAMA(model_path, 16)

# 初始化一个字典来存储每个任务的准确率
accuracies = {}

for data_task in glue_tasks:
    print(f"正在处理任务：{data_task}")
    data = datas[data_task]  # 获取该任务的数据列表，每个元素是一个包含'text'和'label'的字典
    chat_template = template[data_task]  # 获取该任务的模板

    labels = []
    predictions = []
    start_time = time.time()
    for i, item in enumerate(data):
        text = item['text']
        label = item['label']
        labels.append(label)
        # 格式化提示
        prompt = text
        # 运行模型
        output = model.run(chat_template, prompt)
        # 从输出字符串的头开始查找第一个'0'、'1'或'2'
        pred_label = -1  # 默认值，表示未找到有效的预测
        for ch in output:
            if ch == '0':
                pred_label = 0
                break
            elif ch == '1':
                pred_label = 1
                break
            elif ch == '2':
                pred_label = 2
                break
            elif ch in [' ', '\n', '\t']:
                # 跳过空白字符
                continue
            else:
                # 遇到非数字字符，继续查找
                continue
        print(pred_label)
        predictions.append(pred_label)
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"已处理 {i+1} 个样本，耗时 {elapsed:.2f} 秒")
    # 计算准确率
    labels = labels[:len(predictions)]
    labels = np.array(labels)
    predictions = np.array(predictions)
    correct = (predictions == labels)
    accuracy = np.mean(correct)
    accuracies[data_task] = accuracy
    print(f"{data_task} 的准确率：{accuracy*100:.2f}%")

# 打印所有任务的准确率
print("\n所有任务的准确率：")
for task, acc in accuracies.items():
    print(f"{task}: {acc*100:.2f}%")