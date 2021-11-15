import ast
import random
import os
import editdistance
import argparse
import copy
import re
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="path of correction file")
    parser.add_argument("-gold", help="path of gold file")
    parser.add_argument("-noise", help="path of noisy file")

    args = parser.parse_args()

    with open(args.f, 'r') as f:
        correction_lines = f.readlines()

    with open(args.gold, 'r') as f:
        gold_lines = f.readlines()

    with open(args.noise, 'r') as f:
        noisy_lines = f.readlines()

    assert len(gold_lines) == len(noisy_lines)
    assert len(correction_lines) == len(gold_lines)

    correct = 0
    total = 0

    for cor, gold, noisy in zip(correction_lines, gold_lines, noisy_lines):
        cor = cor.strip()
        gold = gold.strip()
        noisy = noisy.strip()

        equal = True

        cor_words = cor.split()
        gold_words = gold.split()
        noisy_words = noisy.split()

        for i, (noisy_word, gold_word) in enumerate(zip(noisy_words, gold_words)):
            if noisy_word[0] == '[' and noisy_word[-1] == ']':
                if gold_word in cor:
                    correct += 1
                total += 1
    
    print("Correction Acc:", correct / total)