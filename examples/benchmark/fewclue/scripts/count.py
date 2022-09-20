from utils import load_fewclue, LABEL_MAP


def count_max_length(task_name):
    max_length = 0
    all_length = []
    for i in range(5):
        train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
            task_name, i, LABEL_MAP[task_name])

        for example in train_ds:
            all_length.append(len(example.text_a) + len(example.text_b))
            max_length = max(max_length,
                             len(example.text_a) + len(example.text_b))

        for example in dev_ds:
            all_length.append(len(example.text_a) + len(example.text_b))
            max_length = max(max_length,
                             len(example.text_a) + len(example.text_b))

        for example in public_test_ds:
            all_length.append(len(example.text_a) + len(example.text_b))
            max_length = max(max_length,
                             len(example.text_a) + len(example.text_b))

        for example in test_ds:
            all_length.append(len(example.text_a) + len(example.text_b))
            max_length = max(max_length,
                             len(example.text_a) + len(example.text_b))
    return max_length, sum(all_length) / len(all_length)


dataset = list(LABEL_MAP.keys())

for name in dataset:
    print(name)
    max_length, avg_length = count_max_length(name)
    print(name, ": max_length:", max_length, ", avg_length:", avg_length)
