from utils import *

print("加载 PTB (Penn Treebank) 数据集...")
ptb = load_dataset("datasets/ptb.py", "penn_treebank", cache_dir=datasets_dir, download_config=download_config, trust_remote_code=True)

# 2. 在 PTB 数据集上计算困惑度
print("\n=== 在 PTB 数据集上计算困惑度 ===")

ptb_test = ptb['test']

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

ptb_encoded = ptb_test.map(preprocess_ptb, batched=True, remove_columns=['sentence'])
ptb_dataloader = DataLoader(ptb_encoded, batch_size=2, collate_fn=default_data_collator)

ptb_perplexity = compute_perplexity(model, ptb_dataloader)
print(f"PTB 数据集的困惑度：{ptb_perplexity}")