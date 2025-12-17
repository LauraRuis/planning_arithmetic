from datasets import Dataset


def get_dataset(args, tokenizer):
    dataset = Dataset.from_dict({
        "chat": [
            [
                {"role": "user",
                    "content": "Hello, how are you?"},
                {"role": "assistant",
                    "content": "I'm fine, thank you!"}],
            [
                {"role": "user",
                "content": "What's the weather like today?"},
                {"role": "assistant",
                "content": "It's sunny and warm."}]
        ],
        "example_id": [0, 1]
    })
    if args.dataset_pars.apply_chat_template: 
        dataset = dataset.map(lambda x: {"prompt": tokenizer.apply_chat_template([x["chat"][0]], tokenize=False, add_generation_prompt=False),
                                            "completion": tokenizer.apply_chat_template([x["chat"][1]], tokenize=False, add_generation_prompt=True)})
    else:
        dataset = dataset.map(lambda x: {"prompt": x["chat"][0]["content"],
                                         "completion": x["chat"][1]["content"]})
    return dataset

def get_datasets(args, tokenizer):
    dataset = get_dataset(args, tokenizer)
    # For simplicity, use the same dataset for training and evaluation
    return dataset, dataset