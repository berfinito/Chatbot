import os
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig # Import for quantization
)
# Ensure peft is installed: pip install peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import Accelerator # Handles device placement
import math # For calculating steps if needed

# === CONFIG ===
# --- Data ---
TEXT_FILE = "../full_university_data_dapt.txt"
TOKENIZED_CACHE = "tokenized_full_university_cache_sep_fix" # New cache name
ENTRY_SEPARATOR = "\n\n--- SOURCE SEPARATOR ---\n\n"

# --- Model ---
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR = "dapt_full_dataset_qlora_output"
MERGED_OUTPUT_DIR = "dapt_full_dataset_qlora_merged"

# --- Tokenization & Chunking ---
MAX_SEQ_LENGTH = 512
STRIDE = 256 # Overlap = MAX_SEQ_LENGTH - STRIDE

# --- QLoRA / Training ---
USE_QLORA = True
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 2e-4
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
FP16 = False
BF16 = torch.cuda.is_bf16_supported()

# --- LoRA Specific Config ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Processing ---
NUM_PROC = os.cpu_count() // 2 if os.cpu_count() else 4

# --- Optional Merge Flag ---
SAVE_MERGED_MODEL = True

# === HELPER FUNCTIONS ===

def tokenize_function(examples, tokenizer, max_length, stride):
    """Tokenizes text from documents, creating overlapping chunks."""
    # Set padding side to right for consistency
    tokenizer.padding_side = "right"
    # Tokenize the batch of texts
    tokenized_output = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding=False, # Do not pad here, collator handles batch padding
        return_tensors=None, # Let map handle formatting
    )
    # Important: Remove the 'overflow_to_sample_mapping' column as it can interfere
    # with the data collator if not handled explicitly later.
    # We only need input_ids and attention_mask for the LM collator.
    if "overflow_to_sample_mapping" in tokenized_output:
        del tokenized_output["overflow_to_sample_mapping"]

    return tokenized_output

def load_and_process_data_separated(text_file, cache_dir, tokenizer, max_length, stride, num_proc, separator):
    """Loads data, splits by separator, tokenizes each part respecting boundaries."""
    if os.path.exists(cache_dir):
        print(f"🔁 Using cached tokenized dataset from '{cache_dir}'")
        try:
            ds = load_from_disk(cache_dir)
            if "input_ids" in ds.column_names and "attention_mask" in ds.column_names:
                 print(f"✅ Cache valid. Loaded {len(ds)} sequences.")
                 return ds
            else:
                 print("⚠️ Cache invalid (missing columns). Reprocessing...")
        except Exception as e:
            print(f"⚠️ Failed to load from cache '{cache_dir}': {e}. Reprocessing...")

    print(f"💾 Cache not found or invalid. Processing data from '{text_file}'...")
    if not os.path.exists(text_file):
        print(f"❌ Error: Text file '{text_file}' not found.")
        exit(1)

    # 1. Load Full Text
    print(f"📄 Loading document: {text_file}")
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            full_text = f.read()
        print(f"✅ Document loaded. Total characters: {len(full_text)}")
    except Exception as e:
        print(f"❌ Error reading text file: {e}")
        exit(1)

    # 2. Split into Documents by Separator
    print(f"✂️ Splitting text into documents using separator: '{separator.strip()}'")
    documents = full_text.split(separator)
    documents = [doc.strip() for doc in documents if doc.strip()]
    print(f"✅ Split into {len(documents)} source documents.")
    if not documents:
        print("❌ Error: No documents found after splitting by separator.")
        exit(1)

    # 3. Create a Dataset from documents
    raw_dataset = Dataset.from_dict({"text": documents})
    print(f"📊 Created raw dataset with {len(raw_dataset)} rows (documents).")

    # 4. Tokenize Each Document into Chunks using map
    print(f"⚙️ Tokenizing each document into sequences (using {num_proc} processes)...")
    # Use batched=True for efficiency with multiple documents
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True, # Process multiple documents in batches
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length, "stride": stride},
        num_proc=num_proc,
        remove_columns=["text"] # Remove original document text column
    )

    print(f"✅ Dataset tokenized into {len(tokenized_dataset)} sequences of max_length <= {max_length}.")

    # DO NOT add labels here. Let DataCollatorForLanguageModeling handle it.
    # It automatically creates labels from input_ids for Causal LM.

    print(f"💾 Saving processed dataset to '{cache_dir}'...")
    tokenized_dataset.save_to_disk(cache_dir)
    print(f"✅ Processed dataset saved.")
    return tokenized_dataset

# === MAIN SCRIPT ===
if __name__ == "__main__":
    print("🚀 Starting DAPT Training Script for Full Dataset (QLoRA - Separator Aware)...")

    # --- 1. Load Tokenizer ---
    print(f"🔄 Loading Tokenizer for '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right" # Define padding side early
    if tokenizer.pad_token is None:
        print("⚠️ Tokenizer does not have a pad token. Setting pad_token=eos_token.")
        tokenizer.pad_token = tokenizer.eos_token


    # --- 2. Load and Prepare Data ---
    dataset = load_and_process_data_separated(
        TEXT_FILE,
        TOKENIZED_CACHE,
        tokenizer,
        MAX_SEQ_LENGTH,
        STRIDE,
        NUM_PROC,
        ENTRY_SEPARATOR # Pass the separator
    )
    print(f"📊 Using dataset with {len(dataset)} examples (sequences) for DAPT.")
    if len(dataset) == 0:
        print("❌ Dataset is empty after tokenization. Check data or tokenization parameters.")
        exit(1)

    # --- 3. Configure Quantization ---
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
    )

    # --- 5. Prepare for LoRA ---
    lora_config = None # Initialize
    if USE_QLORA:
        print("🔧 Preparing model for k-bit training and applying LoRA...")
        # Enable gradient checkpointing when preparing the model
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("📊 Model layers after applying LoRA:")
        model.print_trainable_parameters()

    # --- 6. Configure Training ---
    # Data collator for language modeling. Handles padding within batches AND creates labels.
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
        fp16=False, # Handled by QLoRA config
        bf16=BF16, # Use bf16 if available and desired
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # dataloader_num_workers=NUM_PROC // 2 if NUM_PROC > 1 else 0, # Optional
        report_to="none",
        gradient_checkpointing=True, # Enabled via prepare_model_for_kbit_training
        gradient_checkpointing_kwargs={'use_reentrant': False}, # Recommended for QLoRA
    )

    # --- 7. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator, # *** Ensure the collator is passed ***
        tokenizer=tokenizer, # Pass tokenizer for saving purposes
    )

    # --- 8. Start Training ---
    print(f"🚀 Starting DAPT training on full dataset ({'QLoRA' if USE_QLORA else 'Full Parameter'})...")
    try:
        train_result = trainer.train()
        print("✅ Full Dataset DAPT training complete!")

        # --- 9. Save Final Adapters & Stats ---
        print(f"💾 Saving final DAPT model adapters to {OUTPUT_DIR}...")
        trainer.save_model(OUTPUT_DIR) # Saves adapters and config if PEFT, else full model

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("📊 Training metrics saved.")

        # --- 10. Optional: Merge and Save Full Model ---
        if SAVE_MERGED_MODEL and USE_QLORA:
            # (Merge logic remains the same)
            print("\nMerging trained LoRA adapters into the base model...")
            print("⚠️ This requires significant RAM/VRAM and time.")
            try:
                print("Moving model to CPU for merging...")
                trainer.model.to('cpu')
                torch.cuda.empty_cache()

                print(f"Reloading base model ({BASE_MODEL_NAME}) in fp16 on CPU for merging...")
                base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                ).to('cpu')

                print(f"Loading adapters from {OUTPUT_DIR} to merge...")
                merged_model = PeftModel.from_pretrained(
                    base_model_for_merge,
                    OUTPUT_DIR
                )
                print("Unloading and merging...")
                merged_model = merged_model.merge_and_unload()
                print("✅ Merge complete.")

                print(f"💾 Saving merged model to {MERGED_OUTPUT_DIR}...")
                os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
                merged_model.save_pretrained(MERGED_OUTPUT_DIR)
                tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
                print(f"✅ Merged model saved to {MERGED_OUTPUT_DIR}")

            except Exception as merge_error:
                print(f"❌ Error during model merging/saving: {merge_error}")
                print("Skipping merged model saving.")
            finally:
                if 'merged_model' in locals(): del merged_model
                if 'base_model_for_merge' in locals(): del base_model_for_merge

    except Exception as e:
        print(f"❌ An error occurred during DAPT training: {e}")
        import traceback
        traceback.print_exc()

    print("🏁 Full Dataset DAPT script finished.")

