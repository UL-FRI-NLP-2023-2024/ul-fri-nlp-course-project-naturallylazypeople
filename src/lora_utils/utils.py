from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_lora_model(model):
    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_lin", "k_lin","v_lin"],  # The modules (for example, attention blocks) to apply the LoRA update matrices.
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=None,  # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
        task_type="SEQ_CLS"
    )
    lora_model = get_peft_model(model, config)
    return lora_model

def get_lora_training_args(model_name: str):
    training_args = TrainingArguments(
        output_dir=f"{model_name}-scene-parse-150-lora",
        learning_rate=5e-4,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    return training_args

def lora_trainable(lora_model):
    for name, param in lora_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

def get_lora_trainer(lora_model, training_args, train_dataset, eval_dataset, compute_metrics=None):
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer

# print_trainable_parameters(lora_model)