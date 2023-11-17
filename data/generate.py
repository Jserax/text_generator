import torch
from transformers import AutoTokenizer


def generate(
    input_path: str,
    output_path: str,
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepPavlov/distilrubert-tiny-cased-conversational-5k"
    )
    tensor = tokenizer.encode(
        data, add_special_tokens=False, return_tensors="pt"
    ).squeeze()
    print(len(tensor))
    torch.save(tensor, output_path)


if __name__ == "__main__":
    generate("data/Gazdanov (1).txt", "data/data.pt")
