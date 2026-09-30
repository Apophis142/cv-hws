import argparse
import os
from collections import defaultdict

CLASSES = ['red', 'yellow', 'green']


def read_classes(directory):
    mapping = {}
    for cls in CLASSES:
        path = os.path.join(directory, f'{cls}.txt')
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    mapping[name] = cls
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",
                        default="tasks/traffic_lights_classifier/answers/", help="Directory with predicted answer")
    parser.add_argument("--true",
                        default="data/traffic_lights/answers/", help="Directory with correct answer")
    parser.add_argument("--details", action="store_true", help="List all objects with wrong class")
    args = parser.parse_args()

    pred = read_classes(args.pred)
    true = read_classes(args.true)

    all_names = set(pred) | set(true)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    discrepancies = []

    for name in all_names:
        p = pred.get(name)
        t = true.get(name)

        if p == t:
            correct += 1
            if p is not None:
                tp[p] += 1
        else:
            if p is not None:
                fp[p] += 1
            if t is not None:
                fn[t] += 1
            discrepancies.append((name, p, t))

    total = len(all_names)
    accuracy = correct / total if total > 0 else 0.0

    f1_per_class = {}
    for cls in CLASSES:
        tp_c = tp[cls]
        fp_c = fp[cls]
        fn_c = fn[cls]

        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class[cls] = f1

    f1_macro = sum(f1_per_class.values()) / len(CLASSES)

    sum_tp = sum(tp.values())
    sum_fp = sum(fp.values())
    sum_fn = sum(fn.values())

    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    f1_micro = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-micro: {f1_micro:.4f}")

    if args.details:
        print("\nРасхождения (файл: предсказано -> истинно):")
        for name, p, t in sorted(discrepancies):
            print(f"{name}: {p} -> {t}")


if __name__ == '__main__':
    main()
