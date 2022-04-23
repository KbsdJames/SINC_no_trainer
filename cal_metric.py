import pdb
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import Counter
from numpy import mean


def ngrams(sequence, n):
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def compute_distinct_n(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(iter(sentence), n))
    return len(distinct_ngrams) / len(sentence)


def compute_f1(pred_items, gold_items):
    common = Counter(pred_items) & Counter(gold_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_BLEU_batch(preds, refs):
    BLEU_1 = []
    BLEU_2 = []
    for i in range(len(preds)):
        BLEU_1.append(sentence_bleu([refs[i]], preds[i], weights=(1, 0, 0, 0)))
        BLEU_2.append(sentence_bleu([refs[i]], preds[i], weights=(0, 1, 0, 0)))
    return mean(BLEU_1), mean(BLEU_2)

def compute_f1_batch(preds, refs):
    f1 = []
    for i in range(len(preds)):
        f1.append(compute_f1(preds[i], refs[i]))
    return mean(f1)

def compute_distinct_batch(preds):
    dis_1 = []
    dis_2 = []
    for i in range(len(preds)):
        dis_1.append(compute_distinct_n(preds[i], 1))
        dis_2.append(compute_distinct_n(preds[i], 2))
    return mean(dis_1), mean(dis_2)