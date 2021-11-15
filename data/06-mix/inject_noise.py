import ast
import random
import os
import editdistance
import argparse
import copy
import re
import numpy as np
import pronouncing
from scipy import spatial
from tqdm import tqdm

random.seed(1)
pronouncing.init_cmu()
pronunciations = pronouncing.pronunciations


adjacent_keys = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 's'],
    'e': ['w', 'r', 'd'],
    'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'],
    'y': ['t', 'u', 'h'],
    'u': ['y', 'i', 'j'],
    'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 's', 'z'],
    's': ['w', 'a', 'd', 'x', 'z'],
    'd': ['e', 's', 'f', 'c', 'x'],
    'f': ['d', 'r', 'g', 'v', 'c'],
    'g': ['f', 't', 'h', 'v', 'b'],
    'h': ['g', 'y', 'b', 'n', 'j'],
    'j': ['h', 'u', 'n', 'm', 'k'],
    'k': ['j', 'i', 'm', 'l'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}

filename = [
    '../00-tok/test_2016_flickr.lc.norm.tok.en',
    '../00-tok/test_2017_flickr.lc.norm.tok.en',
    '../00-tok/test_2017_mscoco.lc.norm.tok.en',
    # '../00-tok/train+val.lc.norm.tok.en'
]

vocab_file = '../00-tok/train+val.lc.norm.tok.vocab.en'

# helper func
def find_noisy_editdistance_word(word, vocab, threshold=2):
    random.shuffle(vocab)
    for candidate in vocab:
        if candidate != word and editdistance.eval(candidate, word) <= threshold:
            return candidate, True
    return word, False

# helper func
def find_homophones(word):
    phones = pronouncing.phones_for_word(word)
    
    noisy_word = word

    exact_match = []
    fuzzy_match = []
    for phone in phones:
        for candidate, candidate_phone in pronunciations:
            if phone == candidate_phone:
                exact_match.append(candidate)
            elif candidate_phone in phone and len(candidate_phone.split()) + 1 == len(phone.split()) or \
                 phone in candidate and len(candidate_phone.split()) - 1 == len(phone.split()):
                fuzzy_match.append(candidate)
    
    if phones:
        exact_match.remove(word)
    if exact_match:
        noisy_word = random.choice(exact_match)
    elif fuzzy_match:
        noisy_word = random.choice(fuzzy_match)

    return noisy_word

def inject_editdistance_noise(line, n, lc, vocab):
    words = line.strip().split()
    words_for_indication = words.copy()

    constrained_pos = pos_met_constraint = [i for i, word in enumerate(words) if len(word) >= lc and word.isalpha()]
    random.shuffle(constrained_pos)

    count = 0
    for pos in constrained_pos:
        if count >= n:
            break
        current_word = words[pos]
        noisy_word, found_noisy_word = find_noisy_editdistance_word(current_word, vocab)
        if found_noisy_word:
            count += 1
            words[pos] = noisy_word
            words_for_indication[pos] = '[' + noisy_word + ']'

    noisy_line = ' '.join(words) + '\n'
    indication_line = ' '.join(words_for_indication) + '\n'
    return noisy_line, indication_line, count

def inject_keyboard_nosie(line, n, lc):
    words = line.strip().split()
    words_for_indication = words.copy()

    constrained_pos = pos_met_constraint = [i for i, word in enumerate(words) if len(word) >= lc and word.isalpha()]
    random.shuffle(constrained_pos)

    count = 0
    for pos in constrained_pos:
        if count >= n:
            break
        current_word = words[pos]

        num_chars = len(current_word)
        char_pos = random.choice(range(num_chars))
        char_to_replace = current_word[char_pos]

        replaced_char = random.choice(adjacent_keys[char_to_replace])

        noisy_word = current_word[:char_pos] + replaced_char + current_word[char_pos+1:]
        indication_word = '[' + noisy_word + ']'
        count += 1

        words[pos] = noisy_word
        words_for_indication[pos] = indication_word

    noisy_line = ' '.join(words) + '\n'
    indication_line = ' '.join(words_for_indication) + '\n'
    return noisy_line, indication_line, count

def inject_homophone_noise(line, n, lc):
    words = line.strip().split()
    words_for_indication = words.copy()

    constrained_pos = pos_met_constraint = [i for i, word in enumerate(words) if len(word) >= lc and word.isalpha()]
    random.shuffle(constrained_pos)

    count = 0
    for pos in constrained_pos:
        if count >= n:
            break
        current_word = words[pos]
        
        noisy_word = find_homophones(current_word)
        indication_word = '[' + noisy_word + ']'

        if noisy_word != current_word:
            count += 1

        words[pos] = noisy_word
        words_for_indication[pos] = indication_word

    noisy_line = ' '.join(words) + '\n'
    indication_line = ' '.join(words_for_indication) + '\n'
    return noisy_line, indication_line, count

def inject_noise(lines, n, vocab, length_constraint=2):
    noisy_lines = []
    indication_lines = []
    changed_word = 0

    for line in tqdm(lines):
        noise_type = random.choice(['editdistance', 'keyboard', 'homophone'])

        if noise_type == 'editdistance':
            noisy_line, indication_line, changes = inject_editdistance_noise(line, n, length_constraint, vocab)
        elif noise_type == 'keyboard':
            noisy_line, indication_line, changes = inject_keyboard_nosie(line, n, length_constraint)
        elif noise_type == 'homophone':
            noisy_line, indication_line, changes = inject_homophone_noise(line, n, length_constraint)
        
        noisy_lines.append(noisy_line)
        indication_lines.append(indication_line)
        changed_word += changes

    print("In total {} words changed, average {:.2f} words per sentence".format(changed_word, changed_word / len(lines)))
    return noisy_lines, indication_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of words to be replaced", type=int)
    parser.add_argument("-lc", help="words are changed only if the has more than lc characters", type=int, default=3)
    args = parser.parse_args()

    with open(vocab_file, 'r') as f:
        vocab = ast.literal_eval(f.read()).keys()
        vocab = list(vocab)


    out_dir = "noise_word=" + str(args.n) + "-lconstrain=" + str(args.lc) +"/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fname in filename:
        # read lines from each file
        with open(fname, 'r') as f:
            lines = f.readlines()
        
        noisy_lines, indication_lines = inject_noise(lines, args.n, vocab, args.lc)


        with open(out_dir + fname.split('/')[-1], 'w') as f:
            f.writelines(noisy_lines)

        with open(out_dir + fname.split('/')[-1] + '.marked', 'w') as f:
            f.writelines(indication_lines)



