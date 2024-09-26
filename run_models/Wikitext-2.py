from utils import *

print("加载 Wikitext-2 数据集...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=datasets_dir, download_config=download_config)

print("\n=== 在 Wikitext-2 数据集上计算困惑度 ===")

wikitext_test = wikitext['test']

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

wikitext_encoded = wikitext_test.map(preprocess_wikitext, batched=True, remove_columns=['text'])
wikitext_dataloader = DataLoader(wikitext_encoded, batch_size=2, collate_fn=default_data_collator)

wikitext_perplexity = compute_perplexity(model, wikitext_dataloader)
print(f"Wikitext-2 数据集的困惑度：{wikitext_perplexity}")