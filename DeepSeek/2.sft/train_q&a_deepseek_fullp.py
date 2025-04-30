import os
import json
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig 
)

from trl import SFTTrainer
from accelerate import Accelerator

# === CONFIG ===

# --- Model ---
MODEL_TO_TRAIN_PATH = "../1.dapt/dapt_deepseek_1.3b_model_output"
OUTPUT_DIR = "sft_dapt_1.3b_fp32_output"
ORIGINAL_BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print(f"*** CONFIG: Fine-tuning DAPT 1.3B model (FP32): {MODEL_TO_TRAIN_PATH} ***")
if not os.path.exists(MODEL_TO_TRAIN_PATH):
     print(f"⚠️ WARNING: Starting model path not found: {MODEL_TO_TRAIN_PATH}. Cannot run SFT.")
     exit(1)

# --- Data ---
JSON_DATA_FILE = "health_safety_training_data.json"
PROCESSED_DATA_CACHE = f"processed_sft_qa_1.3b_cache" 


USE_QLORA = False

MAX_SEQ_LENGTH = 512 
EPOCHS = 3 
BATCH_SIZE = 1 
GRAD_ACCUM = 16 
LEARNING_RATE = 5e-5 
OPTIMIZER = "adamw_torch" 
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
FP16 = False
BF16 = False

# --- Multi-processing ---
NUM_PROC = os.cpu_count() // 2 if os.cpu_count() else 4

# === HELPER FUNCTION ===
def format_qa_data(example, tokenizer):
    """
    Formats a single Q&A example from the JSON into the chat template
    expected by the tokenizer. Uses the tokenizer's chat template.
    """
    try:
        if 'messages' not in example or not isinstance(example['messages'], list):
            print(f"Skipping invalid example format: {example}")
            return {"text": None}

        if tokenizer.chat_template is None:
             print("ERROR: Chat template not set on tokenizer before calling format_qa_data!")
             return {"text": None}

        formatted_text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False 
        )
        return {"text": formatted_text}
    except Exception as e:
        print(f"Error formatting example: {example}")
        print(f"Error during apply_chat_template: {e}")
        return {"text": None}


# === MAIN SCRIPT ===
if __name__ == "__main__":
    print(f"🚀 Starting SFT Fine-tuning Script for {MODEL_TO_TRAIN_PATH} (FP32)...")

    # --- 1. Load and Prepare Data ---
    print(f"💾 Loading Q&A data from '{JSON_DATA_FILE}'...")
    try:
        raw_dataset = load_dataset("json", data_files=JSON_DATA_FILE, split="train")
        raw_dataset = raw_dataset.filter(lambda x: x.get('messages') is not None and isinstance(x.get('messages'), list) and len(x['messages']) > 0)
        print(f"📊 Loaded {len(raw_dataset)} Q&A pairs.")
    except Exception as e:
        print(f"❌ Error loading JSON data: {e}")
        exit(1)

    # --- 2. Load Tokenizer ---
    print(f"🔄 Loading Tokenizer from '{MODEL_TO_TRAIN_PATH}' (or fallback to base)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_TRAIN_PATH, trust_remote_code=True)
    except OSError:
        print(f"⚠️ Tokenizer not found in {MODEL_TO_TRAIN_PATH}, loading from {ORIGINAL_BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_BASE_MODEL_NAME, trust_remote_code=True)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            print("⚠️ Tokenizer does not have a pad token. Setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("⚠️ Tokenizer has no pad_token or eos_token. Adding a default pad token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if tokenizer.chat_template is None:
        print("⚠️ Tokenizer does not have a default chat template. Setting a generic one.")
        template_jinja = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "{{ bos_token + message['content'] + '\\n' }}"
                "{% elif message['role'] == 'user' %}"
                    "{{ bos_token + 'USER: ' + message['content'] + '\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ 'ASSISTANT: ' + message['content'] + eos_token + '\\n' }}"
                "{% else %}"
                     "{{ bos_token + message['role'] + ': ' + message['content'] + '\\n' }}"
                "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = template_jinja
        print(f"✅ Chat template set to: {tokenizer.chat_template}")


    # --- 3. Format Data using Multi-processing ---
    print(f"⚙️ Formatting dataset using tokenizer's chat template (using {NUM_PROC} processes)...")
    def check_cache(cache_path):
        if os.path.exists(cache_path):
            try:
                _ = load_from_disk(cache_path, keep_in_memory=False)
                print(f"🔁 Using cached processed dataset from '{cache_path}'")
                return True
            except Exception as e:
                print(f"⚠️ Cache found but seems invalid at '{cache_path}': {e}. Reprocessing...")
                return False
        return False

    if check_cache(PROCESSED_DATA_CACHE):
         formatted_dataset = Dataset.load_from_disk(PROCESSED_DATA_CACHE)
    else:
        print(f"💾 Cache not found or invalid. Processing dataset...")
        formatted_dataset = raw_dataset.map(
            format_qa_data,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=NUM_PROC,
            remove_columns=[col for col in raw_dataset.column_names if col != 'messages']
        )
        num_before_filter = len(formatted_dataset)
        formatted_dataset = formatted_dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
        num_after_filter = len(formatted_dataset)
        if num_before_filter > num_after_filter:
            print(f"⚠️ Filtered out {num_before_filter - num_after_filter} examples that failed formatting.")
        formatted_dataset = formatted_dataset.remove_columns([col for col in formatted_dataset.column_names if col != 'text'])

        print(f"💾 Saving processed dataset to '{PROCESSED_DATA_CACHE}'...")
        os.makedirs(PROCESSED_DATA_CACHE, exist_ok=True)
        formatted_dataset.save_to_disk(PROCESSED_DATA_CACHE)

    if len(formatted_dataset) == 0:
        print("❌ Error: Dataset is empty after processing. Check data formatting or chat template issues.")
        exit(1)

    print(f"📊 Using {len(formatted_dataset)} processed examples for training.")
    print("✨ Example formatted text:")
    print(formatted_dataset[0]['text'])

    # --- 4. Load Model ---
    print(f"🔄 Loading DAPT Model '{MODEL_TO_TRAIN_PATH}' for SFT (FP32)...")
    accelerator = Accelerator()
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_TO_TRAIN_PATH, 
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float32 
    )

    # Resize embeddings if needed
    if tokenizer.pad_token != tokenizer.eos_token and tokenizer.pad_token_id >= model.config.vocab_size:
         print("Resizing token embeddings for added pad token.")
         model.resize_token_embeddings(len(tokenizer))
    elif tokenizer.pad_token_id == tokenizer.eos_token_id and len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
         print("Resizing token embeddings because vocab size changed (likely pad=eos).")
         model.resize_token_embeddings(len(tokenizer))


    # --- 5. Configure Training Arguments ---
    print("📜 Configuring Training Arguments...")
    gradient_checkpointing_enabled = False
    print(f"🧠 Gradient Checkpointing Enabled: {gradient_checkpointing_enabled}")
    if gradient_checkpointing_enabled:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        else:
            print("⚠️ Model does not have gradient_checkpointing_enable method.")
            gradient_checkpointing_enabled = False


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=FP16, 
        bf16=BF16,
        gradient_checkpointing=gradient_checkpointing_enabled,
        report_to="none",
        overwrite_output_dir=True,
    )

    # --- 6. Initialize SFTTrainer ---
    print("🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        tokenizer=tokenizer, 
    )

    # --- 7. Start Training ---
    print(f"🚀 Starting SFT fine-tuning for {MODEL_TO_TRAIN_PATH} (FP32)...")
    try:
        train_result = trainer.train()
        print("✅ SFT Fine-tuning complete!")

        # --- 8. Save Final Model & Stats ---
        print(f"💾 Saving final SFT model (full parameters) to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Model and Tokenizer saved to {OUTPUT_DIR}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("📊 Training metrics saved.")

    except Exception as e:
        print(f"❌ An error occurred during SFT training: {e}")
        import traceback
        traceback.print_exc()


    print(f"🏁 SFT script finished for {MODEL_TO_TRAIN_PATH} (FP32).")

