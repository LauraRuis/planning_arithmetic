from datasets import Dataset

from collections import defaultdict
import numpy as np
import random
import regex
import json

from src.helpers import compute_pass_at_k


def numCarryOps(a, b):
    a,b=int(a),int(b)
    def digitSum(n):
        return sum(map(int,str(n)))
    # assert(a >= 0); assert(b >= 0);
    return int((digitSum(a) + digitSum(b) - digitSum(a+b)) / 9)


class ArithmeticDataset:

    def __init__(self, seed, tokenizer, num_digits, percentage_test, apply_chat_template, balance_carries):
        output_pattern = r"""\\boxed\{
            (?P<content>
                (?:
                    [^{}]
                | (?P<brace>
                        \{
                            (?: [^{}] | (?&brace) )*
                        \}
                    )
                )*
            )
        \}"""
        self.output_pattern = output_pattern
        self.num_digits = num_digits
        self.tokenizer = tokenizer
        self.percentage_test = percentage_test
        self.apply_chat_template = apply_chat_template
        self.seed = seed 
        self.balance_carries = balance_carries
    
    def extract_output(self, completion: str):
        last = None
        for m in regex.finditer(self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE):
            last = m.group('content')
        return last
    
    def extract_output_first(self, completion: str):
        m = regex.search(self.output_pattern, completion, flags=regex.DOTALL | regex.VERBOSE)
        if m:
            return m.group("content")
        return None
    
    def eval_outputs(self, outputs, pass_at_k: int, is_base_model: bool, scratch):

        accuracy = 0
        pass_at_k_scores = {k: 0 for k in range(1, pass_at_k + 1)}
        all_predictions = []

        for i, out in enumerate(outputs):
            print(f"Evaluating output {i+1}/{len(outputs)}")

            # Get the predictions
            texts = [t for t in out["generated_text"]]
            example_id = out["example_id"]
            ground_truth = out['ground_truth']
            preds = []
            num_correct = 0
            
            for text in texts:

                if not scratch:
                    if is_base_model:
                        pred = self.extract_output_first(text)
                    else:
                        pred = self.extract_output(text)
                else:
                    pred = text.strip()
                preds.append(pred)
                if pred is not None and pred == ground_truth[-1]:
                    num_correct += 1
            example_accuracy = num_correct / len(texts)
            accuracy += example_accuracy
            current_pass_at_k = {k: 0 for k in range(1, pass_at_k + 1)}
            for k in range(1, pass_at_k + 1):
                p_at_k = compute_pass_at_k(len(texts), num_correct, k)
                pass_at_k_scores[k] += p_at_k
                current_pass_at_k[k] = p_at_k
            all_predictions.append({
                "example_id": example_id,
                "ground_truth": ground_truth,
                "accuracy": example_accuracy,
                "pass_at_k": {k: p_at_k for k, p_at_k in current_pass_at_k.items()},
                "predictions": preds,
                "prompt": out["prompt"],
                "texts": texts
            })
        accuracy /= len(outputs)
        for k in pass_at_k_scores:
            pass_at_k_scores[k] /= len(outputs)
        return accuracy, pass_at_k_scores, all_predictions
    
    def generate_data(self):
        random.seed(self.seed)
        
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
                if random.random() < self.percentage_test and num_carries >= self.num_digits - 1:
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
    
    def save_state(self, save_path):
        state = {
            "num_digits": self.num_digits,
            "percentage_test": self.percentage_test,
            "apply_chat_template": self.apply_chat_template,
            "seed": self.seed,
            "balance_carries": self.balance_carries
        }
        with open(save_path, "w") as f:
            json.dump(state, f)
    
    @classmethod
    def from_state(cls, state_path, tokenizer):
        with open(state_path, "r") as f:
            state = json.load(f)
        return cls(
            seed=state["seed"],
            tokenizer=tokenizer,
            num_digits=state["num_digits"],
            percentage_test=state["percentage_test"],
            apply_chat_template=state["apply_chat_template"],
            balance_carries=state["balance_carries"]
        )
    