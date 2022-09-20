from utils import load_fewclue, LABEL_MAP


def check_equivalent(task_name):
    print("\n" + "=" * 20 + "\n" + task_name + "\n" + "=" * 20 + "\n")
    train_ds, dev_ds = [], []
    for i in range(5):
        train, dev, _, _ = load_fewclue(task_name, i, LABEL_MAP[task_name])
        train_ds.extend([x.text_a for x in train])
        dev_ds.extend([x.text_a for x in dev])

    train_all, dev_all, _, _ = load_fewclue(task_name, "few_all",
                                            LABEL_MAP[task_name])

    train_all = [x.text_a for x in train_all]
    dev_all = [x.text_a for x in dev_all]

    print("train:")
    print("Sum - All", len(set(train_ds) - set(train_all)),
          set(train_ds) - set(train_all))
    print("All - Sum", len(set(train_all) - set(train_ds)),
          set(train_all) - set(train_ds))
    print("dev:")
    print("Sum - All", len(set(dev_ds) - set(dev_all)),
          set(dev_ds) - set(dev_all))
    print("All - Sum", len(set(dev_all) - set(dev_ds)),
          set(dev_all) - set(dev_ds))


for name in list(LABEL_MAP.keys()):
    check_equivalent(name)
