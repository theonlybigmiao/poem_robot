from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "uer/gpt2-chinese-poem"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from datasets import load_dataset

# 加载自定义文本文件，text.txt 每一行是一段文本
dataset = load_dataset("text", data_files={"train": "/data/poem.txt"})


def tokenize_function(examples):
    encoding = tokenizer(
        examples["text"],
        max_length=70,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
        add_special_tokens=True,
    )
    return encoding


# 应用 tokenizer
tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_test_split = tokenized_datasets["train"].train_test_split(
    test_size=0.1, shuffle=True
)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
