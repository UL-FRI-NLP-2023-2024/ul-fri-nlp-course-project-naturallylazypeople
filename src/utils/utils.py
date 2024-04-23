import re


def clean_text(text):
    # Convert to lowercase
    cleaned_text = text.lower()

    # Remove special characters except whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # Remove extra whitespaces
    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text


def trainable_parameters(model, print=True):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if print:
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    return {"trainable_params": trainable_params, "all_params": all_param, "trainable%": 100 * trainable_params / all_param}