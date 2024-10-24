from utils import *

print("加载 C4：Colossal Clean Crawled Corpus数据集...")
c4 = load_dataset("datasets/c4.py", "en", split="validation", streaming=True, cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True)

# 7. 在 C4数据集上计算困惑度
print("\n=== 在 C4数据集上计算困惑度 ===")

def preprocess_c4(batch):
    inputs = tokenizer(
        batch['text'],
        truncation=True,
        max_length=512,
        return_tensors='pt',
        padding='max_length'
    )
    inputs['labels'] = inputs['input_ids']
    return inputs

c4_encoded = c4.map(preprocess_c4, batched=True)
c4_dataloader = DataLoader(c4_encoded, batch_size=2)

c4_perplexity = compute_perplexity(model, c4_dataloader)
print(f"C4 数据集的困惑度：{c4_perplexity}")