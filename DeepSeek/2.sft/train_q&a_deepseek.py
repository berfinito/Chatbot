import os
import json
import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig # For potential quantization
)
# Ensure peft is installed: pip install peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
# Ensure trl is installed: pip install trl
from trl import SFTTrainer
from accelerate import Accelerator # Handles device placement

# === CONFIG ===

# --- CHOOSE WHICH MODEL TO FINE-TUNE ---
MODEL_CHOICE = "DAPT" # CHANGE THIS to "BASE" or "DAPT" for each run

# --- Define Paths based on Choice ---
ORIGINAL_BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Path to your DAPT model output (containing adapters)
DAPT_ADAPTER_PATH = "../1.dapt/dapt_qlora_model_output"

if MODEL_CHOICE == "BASE":
    MODEL_TO_TRAIN_PATH = ORIGINAL_BASE_MODEL_NAME
    OUTPUT_DIR = "sft_base_8b_qlora_output" # Output dir for SFT on BASE
    print(f"*** CONFIG: Fine-tuning BASE model: {MODEL_TO_TRAIN_PATH} ***")
elif MODEL_CHOICE == "DAPT":
    # For DAPT, we load the base model and apply the DAPT adapters *before* SFT training
    MODEL_TO_TRAIN_PATH = ORIGINAL_BASE_MODEL_NAME # Start from base...
    # We will apply DAPT adapters after loading the base model
    print(f"*** CONFIG: Fine-tuning DAPT model (Base: {MODEL_TO_TRAIN_PATH} + DAPT Adapters: {DAPT_ADAPTER_PATH}) ***")
    OUTPUT_DIR = "sft_dapt_8b_qlora_output" # Output dir for SFT on DAPT
    # Check if DAPT adapters exist
    if not os.path.exists(DAPT_ADAPTER_PATH):
         print(f"⚠️ WARNING: DAPT adapter path not found: {DAPT_ADAPTER_PATH}. Cannot run DAPT SFT.")
         # Decide how to handle this - exit or skip?
         exit(1) # Exit if adapters are missing for DAPT run
else:
    raise ValueError("Invalid MODEL_CHOICE. Set to 'BASE' or 'DAPT'.")


# --- Data ---
JSON_DATA_FILE = "health_safety_training_data.json"
# Use different cache for SFT data processing
PROCESSED_DATA_CACHE = f"processed_sft_qa_cache"

# --- Training ---
# SET THIS TO TRUE TO ENABLE 4-BIT QUANTIZATION (QLORA) - Recommended for 8B models
USE_QLORA = True

MAX_SEQ_LENGTH = 512 # Max sequence length for data processing (used if packing=False)
EPOCHS = 3 # SFT often benefits from a few more epochs than DAPT (e.g., 3)
BATCH_SIZE = 1 # Per device batch size - KEEP LOW for QLoRA
GRAD_ACCUM = 8 # Accumulate gradients (Effective batch size = BATCH_SIZE * GRAD_ACCUM)
LEARNING_RATE = 2e-4 # QLoRA often uses slightly higher LR
OPTIMIZER = "paged_adamw_8bit" if USE_QLORA else "adamw_torch" # Memory efficient optimizer for QLoRA
LR_SCHEDULER = "cosine" # Learning rate scheduler type
WARMUP_RATIO = 0.03 # Percentage of steps for warmup
LOGGING_STEPS = 10
SAVE_STEPS = 100 # Save checkpoints periodically
SAVE_TOTAL_LIMIT = 2

# --- LoRA Specific Config (only used if USE_QLORA=True) ---
# These settings define the *new* adapters trained during SFT
LORA_R = 16             # LoRA rank
LORA_ALPHA = 32         # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05     # Dropout for LoRA layers
# Common targets for Llama-style models:
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Multi-processing ---
NUM_PROC = os.cpu_count() // 2 if os.cpu_count() else 4

# --- Quantization Compute Dtype (if USE_QLORA=True) ---
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# === HELPER FUNCTION ===
def format_qa_data(example, tokenizer):
    """
    Formats a single Q&A example from the JSON into the chat template
    expected by the tokenizer. Uses the tokenizer's chat template.
    """
    # The tokenizer.apply_chat_template handles the formatting including special tokens.
    # Set add_generation_prompt=False as we provide the full conversation (user + assistant)
    try:
        # Ensure 'messages' key exists and is a list
        if 'messages' not in example or not isinstance(example['messages'], list):
            print(f"Skipping invalid example format: {example}")
            return {"text": None} # Return None or empty to filter later

        formatted_text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}
    except Exception as e:
        print(f"Error formatting example: {example}")
        print(f"Error: {e}")
        # Return None or empty text to filter out problematic examples
        return {"text": None}


# === MAIN SCRIPT ===
if __name__ == "__main__":
    print(f"🚀 Starting SFT Fine-tuning Script for {MODEL_CHOICE} model...")
    if USE_QLORA:
        print("🔥 QLoRA (4-bit Quantization) Enabled for SFT")

    # --- 1. Load and Prepare Data ---
    print(f"💾 Loading Q&A data from '{JSON_DATA_FILE}'...")
    try:
        # Load dataset directly from JSON file
        raw_dataset = load_dataset("json", data_files=JSON_DATA_FILE, split="train")
        # Filter out potentially problematic empty examples before processing
        raw_dataset = raw_dataset.filter(lambda x: x.get('messages') is not None and isinstance(x.get('messages'), list) and len(x['messages']) > 0)
        print(f"📊 Loaded {len(raw_dataset)} Q&A pairs.")
    except Exception as e:
        print(f"❌ Error loading JSON data: {e}")
        exit(1)

    # --- 2. Load Tokenizer ---
    # Use the tokenizer associated with the original base model for consistency
    print(f"🔄 Loading Tokenizer for '{ORIGINAL_BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_BASE_MODEL_NAME, trust_remote_code=True)

    # Set padding token if it's not already set (common practice)
    # Add padding side 'right' to prevent issues with certain models/architectures
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        print("⚠️ Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token


    # --- 3. Format Data using Multi-processing ---
    print(f"⚙️ Formatting dataset using tokenizer's chat template (using {NUM_PROC} processes)...")
    # Define function to check if cache should be used
    def check_cache(cache_path):
        if os.path.exists(cache_path):
            try:
                # Try loading dataset info to check validity
                _ = load_from_disk(cache_path, keep_in_memory=False)
                print(f"🔁 Using cached processed dataset from '{cache_path}'")
                return True
            except Exception as e:
                print(f"⚠️ Cache found but seems invalid or corrupt at '{cache_path}': {e}. Reprocessing...")
                return False
        return False

    if check_cache(PROCESSED_DATA_CACHE):
         formatted_dataset = Dataset.load_from_disk(PROCESSED_DATA_CACHE)
    else:
        print(f"💾 Cache not found or invalid. Processing dataset...")
        # Apply formatting function using multiple processes
        formatted_dataset = raw_dataset.map(
            format_qa_data,
            fn_kwargs={"tokenizer": tokenizer}, # Pass tokenizer here
            num_proc=NUM_PROC,
            remove_columns=[col for col in raw_dataset.column_names if col != 'messages'] # Keep 'messages' temporarily if needed for filtering
        )
        # Filter out examples that failed formatting (returned None or empty string)
        formatted_dataset = formatted_dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 0)
        # Remove original columns after filtering is done
        formatted_dataset = formatted_dataset.remove_columns([col for col in formatted_dataset.column_names if col != 'text'])

        print(f"💾 Saving processed dataset to '{PROCESSED_DATA_CACHE}'...")
        # Ensure cache directory exists before saving
        os.makedirs(PROCESSED_DATA_CACHE, exist_ok=True)
        formatted_dataset.save_to_disk(PROCESSED_DATA_CACHE)

    if len(formatted_dataset) == 0:
        print("❌ Error: Dataset is empty after processing. Check data formatting and filtering.")
        exit(1)

    print(f"📊 Using {len(formatted_dataset)} processed examples for training.")
    print("✨ Example formatted text:")
    print(formatted_dataset[0]['text']) # Print first example to verify formatting

    # --- 4. Configure Quantization (if USE_QLORA=True) ---
    bnb_config = None
    if USE_QLORA:
        print(f"⚙️ Configuring BitsAndBytes quantization (4-bit, compute dtype: {compute_dtype})...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, # Often recommended
        )

    # --- 5. Load Model ---
    print(f"🔄 Loading Base Model '{MODEL_TO_TRAIN_PATH}' for SFT...")
    # Use Accelerator to handle device placement automatically
    accelerator = Accelerator()
    device_map = "auto" # Let accelerate/transformers handle device mapping

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_TO_TRAIN_PATH, # Load the base model path
        quantization_config=bnb_config if USE_QLORA else None, # Apply quant config if using QLoRA
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=compute_dtype if USE_QLORA else "auto" # Set dtype for QLoRA or let transformers decide
    )

    # Resize embeddings if pad token was added AFTER model init
    if tokenizer.pad_token_id == tokenizer.eos_token_id and len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
         print("Resizing token embeddings as pad_token was set to eos_token and vocab size increased.")
         model.resize_token_embeddings(len(tokenizer))

    # --- 6. Prepare for LoRA / Apply DAPT Adapters ---
    lora_config = None # Initialize lora_config
    if USE_QLORA:
        print("🔧 Preparing model for k-bit training...")
        # Prepare model for k-bit training (gradient checkpointing, layer norm precision)
        # Set use_gradient_checkpointing based on TrainingArguments later if needed
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False) # Set False initially, Trainer handles it

        if MODEL_CHOICE == "DAPT":
            # Load the DAPT adapters onto the base model *before* creating SFT adapters
            print(f"🔄 Applying DAPT LoRA adapters from {DAPT_ADAPTER_PATH}...")
            # Ensure the DAPT adapters are loaded correctly onto the prepared base model
            model = PeftModel.from_pretrained(model, DAPT_ADAPTER_PATH, is_trainable=True)
            print("✅ DAPT adapters applied.")

        # Configure *new* LoRA adapters for the SFT phase
        print("🔧 Configuring NEW LoRA adapters for SFT phase...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Apply SFT LoRA adapters using get_peft_model only if we are NOT loading DAPT adapters
        # If loading DAPT adapters, SFTTrainer will handle applying the new config
        # This avoids applying adapters twice if MODEL_CHOICE == "DAPT"
        if MODEL_CHOICE == "BASE":
             model = get_peft_model(model, lora_config)

        print("📊 Model layers configuration:")
        model.print_trainable_parameters() # Shows % of trainable SFT adapter parameters

    elif MODEL_CHOICE == "DAPT":
        # If not using QLoRA but starting from DAPT adapters, load them here
        print(f"⚠️ Loading DAPT adapters ({DAPT_ADAPTER_PATH}) without QLoRA. Ensure base model matches.")
        model = PeftModel.from_pretrained(model, DAPT_ADAPTER_PATH)


    # --- 7. Configure Training Arguments ---
    print("📜 Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, # Use the dynamically set output dir
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
        fp16=False, # FP16 handled by QLoRA / compute_dtype if USE_QLORA=True
        bf16=False, # BF16 handled by QLoRA / compute_dtype if USE_QLORA=True
        # dataloader_num_workers=NUM_PROC // 2 if NUM_PROC > 1 else 0,
        # Let Trainer handle gradient checkpointing based on the flag
        gradient_checkpointing=USE_QLORA,
        gradient_checkpointing_kwargs={'use_reentrant': False} if USE_QLORA else None, # Recommended for QLoRA
        report_to="none",
        overwrite_output_dir=True,
        # Add max_seq_length here if needed by underlying processes, though SFTTrainer might not use it directly
        # max_seq_length=MAX_SEQ_LENGTH # Usually not needed here for SFTTrainer itself
    )

    # --- 8. Initialize SFTTrainer ---
    print("🏋️ Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model, # Pass the potentially PEFT-modified model
        args=training_args,
        train_dataset=formatted_dataset,
        # eval_dataset=eval_dataset, # Add evaluation dataset if you have one
        # dataset_text_field="text", # REMOVED
        # max_seq_length=MAX_SEQ_LENGTH, # REMOVED
        tokenizer=tokenizer, # Deprecated, but keep for now unless it causes issues
        # packing=PACKING, # REMOVED - Trainer likely handles this differently now
        # Pass LoRA config to SFTTrainer, it handles applying it if model isn't already PeftModel
        # or trains the top-level adapters if it is already PeftModel (like when DAPT adapters loaded)
        peft_config=lora_config if USE_QLORA else None,
    )

    # --- 9. Start Training ---
    print(f"🚀 Starting SFT fine-tuning for {MODEL_CHOICE} model...")
    try:
        train_result = trainer.train()
        print("✅ SFT Fine-tuning complete!")

        # --- 10. Save Final Model & Stats ---
        print(f"💾 Saving final SFT model adapters (or full model if not QLoRA) to {OUTPUT_DIR}...")
        # Saves only the SFT adapters by default if PEFT was used
        trainer.save_model(OUTPUT_DIR)
        # If DAPT adapters were loaded, only the *new* SFT adapters are saved here by default.
        # The original DAPT adapters remain unchanged in their directory.
        print(f"✅ Model/Adapters saved to {OUTPUT_DIR}")

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("📊 Training metrics saved.")

    except Exception as e:
        print(f"❌ An error occurred during SFT training: {e}")
        import traceback
        traceback.print_exc()


    print(f"🏁 SFT script finished for {MODEL_CHOICE} model.")

