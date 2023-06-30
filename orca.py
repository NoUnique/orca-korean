import argparse
import functools
import random
import os
import json
import pandas as pd
from typing import List
from datasets import load_dataset, Dataset, concatenate_datasets


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def niv2_sampling(dataset: Dataset, max_num_samples_per_task: int = 300, random_seed: int = 42):
    tasks = {}
    completed = {}
    dataset = dataset.shuffle(seed=random_seed)
    for i, example in enumerate(dataset, start=1):
        task_name = example["task_name"]
        if task_name not in tasks:
            tasks[task_name] = [example]
        elif not completed.get(task_name, False):
            tasks[task_name].append(example)
            if len(tasks[task_name]) >= max_num_samples_per_task:
                completed[task_name] = True
        if i % 100000 == 0:
            print(f"{i} / {len(dataset)}")
    flattened = [example for task_examples in tasks.values() for example in task_examples]
    sampled_dataset = Dataset.from_pandas(pd.DataFrame(data=flattened))
    return sampled_dataset


def stratified_sampling(dataset: Dataset, num_samples: int, random_seed: int = 42):
    if len(dataset) < num_samples:
        return dataset
    random.seed(random_seed)
    T = {}  # indexes of exampled for each tasks
    for i, example in enumerate(dataset, start=1):
        task_name = example["task_name"]
        idx = i - 1
        if task_name not in T:
            T[task_name] = [idx]
        else:
            T[task_name].append(idx)
        if i % 100000 == 0:
            print(f"{i} / {len(dataset)}")
    for t in T:
        random.shuffle(T[t])  # shuffle to randomly sample a query without replacement from t
    Q = []
    while len(Q) < num_samples:
        t = random.choice(list(T.keys()))
        if len(T[t]) > 0:
            q = dataset[T[t].pop()]
            Q.append(q)
    sampled_dataset = Dataset.from_pandas(pd.DataFrame(data=Q))
    return sampled_dataset


def add_system_message_with_ids(example, system_messages: List[dict], system_message_ids: List[int]):
    system_message_id = random.choice(system_message_ids)
    system_message = system_messages[system_message_id - 1]
    example["system_message_id"] = system_message["id"]
    example["system_message"] = system_message["instruction"]
    return example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixtures", type=str, nargs="+", default=["cot", "niv2", "flan2021", "t0"])
    parser.add_argument("--output-dir", type=str, default=f"{os.path.dirname(os.path.abspath(__file__))}/data/orca")
    args = parser.parse_args()
    return args


def main(args):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    mixtures = args.mixtures

    ds_orca = Dataset.from_pandas(pd.DataFrame())
    for mixture in mixtures:
        dataset_name = f"conceptofmind/{mixture}_submix_original"
        system_messages = read_jsonl(f"{root_dir}/data/system_prompts.jsonl")
        ds = load_dataset(dataset_name, split="train")
        ds = ds.filter(lambda x: x["template_type"].startswith("zs_"))
        if mixture == "cot":
            # no sampling
            ds = ds.add_column("subcollection_name", [mixture] * len(ds))
            system_message_ids = [6, 11, 16]
            add_system_message = functools.partial(
                add_system_message_with_ids,
                system_messages=system_messages,
                system_message_ids=system_message_ids,
            )
            ds = ds.map(add_system_message)

        elif mixture == "niv2":
            ds = niv2_sampling(ds, max_num_samples_per_task=300)
            ds = ds.add_column("subcollection_name", [mixture] * len(ds))
            system_message_ids = [1, 2, 5, 7, 9, 12, 13, 14, 15]
            add_system_message = functools.partial(
                add_system_message_with_ids,
                system_messages=system_messages,
                system_message_ids=system_message_ids,
            )
            ds = ds.map(add_system_message)

        elif mixture == "flan2021":
            ds = stratified_sampling(ds, 2500000)
            ds = ds.add_column("subcollection_name", [mixture] * len(ds))

            ds_mcq = ds.filter(lambda x: x["template_type"] == "zs_opt")
            system_message_ids = [3, 4, 7, 8, 10]
            add_system_message = functools.partial(
                add_system_message_with_ids,
                system_messages=system_messages,
                system_message_ids=system_message_ids,
            )
            ds_mcq = ds_mcq.map(add_system_message)

            ds_not_mcq = ds.filter(lambda x: x["template_type"] == "zs_noopt")
            system_message_ids = [3, 4, 7]
            add_system_message = functools.partial(
                add_system_message_with_ids,
                system_messages=system_messages,
                system_message_ids=system_message_ids,
            )
            ds_not_mcq = ds_not_mcq.map(add_system_message)

            ds = concatenate_datasets([ds_not_mcq, ds_mcq])

        elif mixture == "t0":
            ds = stratified_sampling(ds, 2000000)
            ds = ds.add_column("subcollection_name", [mixture] * len(ds))
            system_message_ids = [1, 2, 3, 5, 7]
            add_system_message = functools.partial(
                add_system_message_with_ids,
                system_messages=system_messages,
                system_message_ids=system_message_ids,
            )
            ds = ds.map(add_system_message)

        ds_orca = concatenate_datasets([ds_orca, ds])
    ds_orca.save_to_disk(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
