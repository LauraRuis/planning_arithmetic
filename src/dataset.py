from datasets import Dataset

from collections import defaultdict
import random


def numCarryOps(a, b):
    a,b=int(a),int(b)
    def digitSum(n):
        return sum(map(int,str(n)))
    # assert(a >= 0); assert(b >= 0);
    return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)


class ArithmeticDataset:

    def __init__(self, seed, tokenizer, num_digits, percentage_test, apply_chat_template, balance_carries):
        self.num_digits = num_digits
        self.tokenizer = tokenizer
        self.percentage_test = percentage_test
        self.apply_chat_template = apply_chat_template
        self.seed = seed 
        self.balance_carries = balance_carries
    
    def eval_outputs(self, outputs):
        ...
    
    def generate_data(self):
        
        train_dataset = {
            "chat": [],
            "example_id": [],
            "ground_truth": []
        }
        test_dataset = {
            "chat": [],
            "example_id": [],
            "ground_truth": []
        }
        max_number = 10 ** self.num_digits - 1
        min_number = 10 ** (self.num_digits - 1)
        example_per_num_carries = defaultdict(list)
        for i in range(min_number, max_number+1):
            for j in range(i, max_number+1):
                num_carries = numCarryOps(i, j)
                example_per_num_carries[num_carries].append((i, j))
        
        example_id = 0
        min_num_carries = min([len(val) for val in example_per_num_carries.values()])
        for num_carries, examples in example_per_num_carries.items():
            if self.balance_carries:
                sampled_examples = random.sample(examples, min_num_carries)
            else:
                sampled_examples = examples
            for (i, j) in sampled_examples:
                question_one = f"{i}+{j}="
                if i != j:
                    question_two = f"{j}+{i}="
                answer = str(i + j)[0]
                message_one = {
                    "chat": [
                        {"role": "user", "content": question_one},
                        {"role": "assistant", "content": answer}
                    ]
                }
                if i != j:
                    message_two = {
                        "chat": [
                            {"role": "user", "content": question_two},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                if random.random() < self.percentage_test:
                    test_dataset["chat"].append(message_one["chat"])
                    test_dataset["example_id"].append(example_id)
                    test_dataset["ground_truth"].append(answer)
                    if i != j:
                        example_id += 1
                        test_dataset["chat"].append(message_two["chat"])
                        test_dataset["example_id"].append(example_id)
                        test_dataset["ground_truth"].append(answer)
                else:
                    train_dataset["chat"].append(message_one["chat"])
                    train_dataset["example_id"].append(example_id)
                    train_dataset["ground_truth"].append(answer)
                    if i != j:
                        example_id += 1
                        train_dataset["chat"].append(message_two["chat"])
                        train_dataset["example_id"].append(example_id)
                        train_dataset["ground_truth"].append(answer)
                example_id += 1
    
        train_dataset = Dataset.from_dict(train_dataset)
        train_dataset = train_dataset.shuffle(seed=self.seed)
        test_dataset = Dataset.from_dict(test_dataset)
        if self.apply_chat_template:
            train_dataset = train_dataset.map(lambda x: {"prompt": self.tokenizer.apply_chat_template([x["chat"][0]], tokenize=False, add_generation_prompt=False),
                                                         "completion": self.tokenizer.apply_chat_template([x["chat"][1]], tokenize=False, add_generation_prompt=True)})
            test_dataset = test_dataset.map(lambda x: {"prompt": self.tokenizer.apply_chat_template([x["chat"][0]], tokenize=False, add_generation_prompt=False),
                                                       "completion": self.tokenizer.apply_chat_template([x["chat"][1]], tokenize=False, add_generation_prompt=True)})
        else:
            train_dataset = train_dataset.map(lambda x: {"prompt": x["chat"][0]["content"],
                                                         "completion": x["chat"][1]["content"]})
            test_dataset = test_dataset.map(lambda x: {"prompt": x["chat"][0]["content"],
                                                       "completion": x["chat"][1]["content"]})
        return train_dataset, test_dataset
    