import os
import sys
import argparse
from typing import List, Tuple

tags = [
    'AJ0',
    'AJC',
    'AJS',
    'AT0',
    'AV0',
    'AVP',
    'AVQ',
    'CJC',
    'CJS',
    'CJT',
    'CRD',
    'DPS',
    'DT0',
    'DTQ',
    'EX0',
    'ITJ',
    'NN0',
    'NN1',
    'NN2',
    'NP0',
    'ORD',
    'PNI',
    'PNP',
    'PNQ',
    'PNX',
    'POS',
    'PRF',
    'PRP',
    'PUL',
    'PUN',
    'PUQ',
    'PUR',
    'TO0',
    'UNC',
    'VBB',
    'VBD',
    'VBG',
    'VBI',
    'VBN',
    'VBZ',
    'VDB',
    'VDD',
    'VDG',
    'VDI',
    'VDN',
    'VDZ',
    'VHB',
    'VHD',
    'VHG',
    'VHI',
    'VHN',
    'VHZ',
    'VM0',
    'VVB',
    'VVD',
    'VVG',
    'VVI',
    'VVN',
    'VVZ',
    'XX0',
    'ZZ0',
    'AJ0-AV0',
    'AJ0-VVN',
    'AJ0-VVD',
    'AJ0-NN1',
    'AJ0-VVG',
    'AVP-PRP',
    'AVQ-CJS',
    'CJS-PRP',
    'CJT-DT0',
    'CRD-PNI',
    'NN1-NP0',
    'NN1-VVB',
    'NN1-VVG',
    'NN2-VVZ',
    'VVD-VVN'
]


def read_training_files(training_files: List[str]) -> List[List[Tuple[str, str]]]:
    """Split training files into sentences, formatted as an array of arrays of words. Words are tuples of the form
    (word, tag).

    Return this array of sentences.
    """
    sentences = []
    for training_file in training_files:
        with open(training_file, 'r') as f:
            curr_sentence = []
            for line in f:
                split = line.split()
                word = (split[0], split[2])
                curr_sentence.append(word)
                if line[0] in ['.', '!', '?']:
                    sentences.append(curr_sentence)
                    curr_sentence = []
    return sentences


def get_initial_probs(sentences: List[List[Tuple[str, str]]]):
    initial_probs = {tag: 0 for tag in tags}
    count = 0
    for sentence in sentences:
        try:
            initial_probs[sentence[0][1]] += 1
        except KeyError:  # ambiguity tag formatted backwards
            split = sentence[0][1].split('-')
            proper_tag = split[1] + '-' + split[0]
            initial_probs[proper_tag] += 1
        count += 1
    for tag in initial_probs:
        initial_probs[tag] /= count
    return initial_probs


def get_observation_probs(sentences: List[List[Tuple[str, str]]]):
    observation_probs = {tag: {} for tag in tags}
    tag_count = {tag: 0 for tag in tags}
    for sentence in sentences:
        for word in sentence:
            try:
                if observation_probs[word[1]].get(word[0]) is None:
                    observation_probs[word[1]][word[0]] = 1
                else:
                    observation_probs[word[1]][word[0]] += 1
                tag_count[word[1]] += 1
            except KeyError:  # ambiguity tag formatted backwards
                split = word[1].split('-')
                proper_tag = split[1] + '-' + split[0]
                if observation_probs[proper_tag].get(word[0]) is None:
                    observation_probs[proper_tag][word[0]] = 1
                else:
                    observation_probs[proper_tag][word[0]] += 1
                tag_count[proper_tag] += 1
    for tag in observation_probs:
        for word in observation_probs[tag]:
            observation_probs[tag][word] /= tag_count[tag]
    return observation_probs, tag_count


def get_transition_probs(sentences, tag_count):
    transition_probs = {tag: {tag: 0 for tag in tags} for tag in tags}
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            if sentence[i + 1][1] in transition_probs and sentence[i][1] in transition_probs[sentence[i + 1][1]]:
                transition_probs[sentence[i + 1][1]][sentence[i][1]] += 1
            elif sentence[i + 1][1] not in transition_probs:
                try:
                    split = sentence[i + 1][1].split('-')
                    proper_tag = split[1] + '-' + split[0]
                    transition_probs[proper_tag][sentence[i][1]] += 1
                except KeyError:
                    split2 = sentence[i][1].split('-')
                    proper_tag2 = split2[1] + '-' + split2[0]
                    transition_probs[proper_tag][proper_tag2] += 1
            elif sentence[i][1] not in transition_probs[sentence[i + 1][1]]:
                split = sentence[i][1].split('-')
                proper_tag = split[1] + '-' + split[0]
                transition_probs[sentence[i + 1][1]][proper_tag] += 1
    for tag1 in transition_probs:
        for tag2 in transition_probs[tag1]:
            transition_probs[tag1][tag2] /= tag_count[tag2]
    return transition_probs


def train(training_files: List[str]):
    """Train the HMM on the given training files."""
    sentences = read_training_files(training_files)
    initial_probs = get_initial_probs(sentences)
    observation_probs, tag_count = get_observation_probs(sentences)
    transition_probs = get_transition_probs(sentences, tag_count)
    return initial_probs, observation_probs, transition_probs




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    print("Starting the tagging process.")
    print(train(training_list))
