from __future__ import annotations

import torch


def confusion_matrix(targets: list[int], predictions: list[int], class_count: int) -> torch.Tensor:
    matrix = torch.zeros((class_count, class_count), dtype=torch.long)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target, prediction] += 1
    return matrix


def macro_f1(targets: list[int], predictions: list[int], class_count: int) -> float:
    matrix = confusion_matrix(targets, predictions, class_count).float()
    scores: list[float] = []
    for class_index in range(class_count):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(float((2 * true_positive / denominator).item()) if denominator > 0 else 0.0)
    return sum(scores) / len(scores)
