import os
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig 
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 
from accelerate import Accelerator

# === CONFIG ===

TEXT_FILE = "../health_and_safety.txt" 
TOKENIZED_CACHE = "tokenized_health_safety_cache" 

# --- Model ---
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR = "dapt_qlora_model_output" 

# --- Tokenization & Chunking ---
MAX_SEQ_LENGTH = 512
CHUNK_OVERLAP = 64

# --- QLoRA / Training ---
USE_QLORA = True

EPOCHS = 1 
BATCH_SIZE = 1 
GRAD_ACCUM = 8 
LEARNING_RATE = 2e-4 
OPTIMIZER = "paged_adamw_8bit" if USE_QLORA else "adamw_torch"
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
SAVE_STEPS = 500 
SAVE_TOTAL_LIMIT = 2 
FP16 = not USE_QLORA
BF16 = False

LORA_R = 16            
LORA_ALPHA = 32        
LORA_DROPOUT = 0.05    

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


NUM_PROC = os.cpu_count() // 2 if os.cpu_count() else 4

# === HELPER FUNCTIONS ===

def tokenize_and_chunk(examples, tokenizer, max_length=512, overlap=64):
    """
    Tokenizes text and chunks it into overlapping sequences.
    Operates on a batch of examples provided by `dataset.map`.
    """
    tokenized_output = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
    )
    input_ids_list = tokenized_output["input_ids"]
    attention_mask_list = tokenized_output["attention_mask"]
    chunked_input_ids = []
    chunked_attention_mask = []
    stride = max_length - overlap

    for doc_input_ids, doc_attention_mask in zip(input_ids_list, attention_mask_list):
        doc_len = len(doc_input_ids)
        start = 0
        while start < doc_len:
            end = min(start + max_length, doc_len)
            chunk_ids = doc_input_ids[start:end]
            chunk_mask = doc_attention_mask[start:end]
            if len(chunk_ids) < max_length:
                padding_length = max_length - len(chunk_ids)
                chunk_ids += [tokenizer.pad_token_id] * padding_length
                chunk_mask += [0] * padding_length
            chunked_input_ids.append(chunk_ids)
            chunked_attention_mask.append(chunk_mask)
            if end == doc_len:
                break
            start += stride
    return {"input_ids": chunked_input_ids, "attention_mask": chunked_attention_mask}


def load_or_process_data(text_file, cache_dir, tokenizer, max_length, overlap, num_proc):
    """Loads data from cache or processes it from the text file."""
    if os.path.exists(cache_dir):
        print(f"🔁 Using cached tokenized dataset from '{cache_dir}'")
        try:
            return load_from_disk(cache_dir)
        except Exception as e:
            print(f"⚠️ Failed to load from cache '{cache_dir}': {e}. Reprocessing...")

    print(f"💾 Cache not found or failed to load. Processing data from '{text_file}'...")
    if not os.path.exists(text_file):
        print(f"❌ Error: Text file '{text_file}' not found.")
        exit(1)

    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    entries = [e.strip() for e in raw_text.split("\n\n") if e.strip()]
    raw_dataset = Dataset.from_dict({"text": entries})

    print(f"⚙️ Tokenizing and chunking dataset (using {num_proc} processes)...")
    tokenized_dataset = raw_dataset.map(
        tokenize_and_chunk,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "overlap": overlap
        },
        num_proc=num_proc,
        remove_columns=["text"]
    )

    print(f"💾 Saving processed dataset to '{cache_dir}'...")
    tokenized_dataset.save_to_disk(cache_dir)
    print(f"✅ Processed dataset saved. Size: {len(tokenized_dataset)} chunks.")
    return tokenized_dataset

# === MAIN SCRIPT ===
if __name__ == "__main__":
    print("🚀 Starting DAPT Training Script...")
    if USE_QLORA:
        print("🔥 QLoRA (4-bit Quantization) Enabled")

    # --- 1. Load Tokenizer ---
    print(f"🔄 Loading Tokenizer for '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token is None:
        print("⚠️ Tokenizer does not have a pad token. Setting pad_token=eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare Data ---
    dataset = load_or_process_data(
        TEXT_FILE,
        TOKENIZED_CACHE,
        tokenizer,
        MAX_SEQ_LENGTH,
        CHUNK_OVERLAP,
        NUM_PROC
    )
    print(f"📊 Using dataset with {len(dataset)} examples.")

    # --- 3. Configure Quantization (if using QLoRA) ---
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16, 
            bnb_4bit_use_double_quant=True, 
        )

    # --- 4. Load Model ---
    print(f"🔄 Loading Model '{BASE_MODEL_NAME}'...")
    accelerator = Accelerator()
    device_map="auto" 

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # --- 5. Prepare for LoRA (if using QLoRA) ---
    if USE_QLORA:
        print("🔧 Preparing model for k-bit training and applying LoRA...")
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        # Apply LoRA adapters to the model
        model = get_peft_model(model, lora_config)
        print("📊 Model layers after applying LoRA:")
        model.print_trainable_parameters() 

    # --- 6. Configure Training ---

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("📜 Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        fp16=FP16,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_num_workers=NUM_PROC // 2 if NUM_PROC > 1 else 0,
        report_to="none",
        gradient_checkpointing=not USE_QLORA,
    )

    # --- 7. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- 8. Start Training ---
    print(f"🚀 Starting DAPT training ({'QLoRA' if USE_QLORA else 'Full Parameter'})...")
    try:
        train_result = trainer.train()
        print("✅ DAPT training complete!")

        # --- 9. Save Final Model & Stats ---
        print(f"💾 Saving final DAPT model adapters (or full model) to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR) 

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("📊 Training metrics saved.")

    except Exception as e:
        print(f"❌ An error occurred during DAPT training: {e}")
        import traceback
        traceback.print_exc() 

    print("🏁 DAPT script finished.")
