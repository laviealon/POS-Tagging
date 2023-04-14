import os
import sys
import argparse
from typing import List, Tuple, Dict

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
    # Calculate probabilities
    initial_probs = {tag: 0 for tag in tags}
    for sentence in sentences:
        try:
            initial_probs[sentence[0][1]] += 1
        except KeyError:  # ambiguity tag formatted backwards
            split = sentence[0][1].split('-')
            proper_tag = split[1] + '-' + split[0]
            initial_probs[proper_tag] += 1
    for tag in initial_probs:
        initial_probs[tag] /= len(sentences)
    # Convert to list for Viterbi
    initial_probs = [initial_probs[tag] for tag in tags]
    return initial_probs


def get_observation_probs(sentences: List[List[Tuple[str, str]]]):
    seen = set()
    observation_probs = {tag: {} for tag in tags}
    tag_count = {tag: 0 for tag in tags}
    for sentence in sentences:
        for word_tag in sentence:
            word, curr_tag = word_tag
            if word not in seen:
                seen.add(word)
                for tag in tags:
                    observation_probs[tag][word] = 0
            try:
                observation_probs[curr_tag][word] += 1
                tag_count[curr_tag] += 1
            except KeyError:  # ambiguity tag formatted backwards
                split = curr_tag.split('-')
                proper_tag = split[1] + '-' + split[0]
                observation_probs[proper_tag][word] += 1
                tag_count[proper_tag] += 1
    for tag in observation_probs:
        for word in observation_probs[tag]:
            try:
                observation_probs[tag][word] /= tag_count[tag]
            except ZeroDivisionError:
                observation_probs[tag][word] = 0
    # convert to list of dicts
    observation_probs = [observation_probs[tag] for tag in tags]
    return observation_probs, tag_count


def get_transition_probs(sentences, tag_count):
    transition_probs = {prev_tag: {tag: 0 for tag in tags} for prev_tag in tags}
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
                    split = sentence[i + 1][1].split('-')
                    proper_tag = split[1] + '-' + split[0]
                    split2 = sentence[i][1].split('-')
                    proper_tag2 = split2[1] + '-' + split2[0]
                    transition_probs[proper_tag][proper_tag2] += 1
            elif sentence[i][1] not in transition_probs[sentence[i + 1][1]]:
                split = sentence[i][1].split('-')
                proper_tag = split[1] + '-' + split[0]
                transition_probs[sentence[i + 1][1]][proper_tag] += 1
    for tag1 in transition_probs:
        for tag2 in transition_probs[tag1]:
            try:
                transition_probs[tag1][tag2] /= tag_count[tag2]
            except ZeroDivisionError:
                transition_probs[tag1][tag2] = 0
    # convert to list of lists. T[j][i] = P(St = tag j | St-1 = tag i), inverted compared to the pseudocode
    transition_probs = [[transition_probs[tag1][tag2] for tag2 in tags] for tag1 in tags]
    return transition_probs


def train(training_files: List[str]):
    """Train the HMM on the given training files."""
    sentences = read_training_files(training_files)
    initial_probs = get_initial_probs(sentences)
    observation_probs, tag_count = get_observation_probs(sentences)
    transition_probs = get_transition_probs(sentences, tag_count)
    return initial_probs, observation_probs, transition_probs


def read_test_file(test_file: str) -> Tuple[List[List[str]], Dict[str, None]]:
    """Read the test file and return a list of sentences, where each sentence is a list of words. Return
    a dictionary of the same words, where the null values will be replaced with the predicted tags.
    """
    sentences = []
    output_dict = {}
    with open(test_file, 'r') as f:
        curr_sentence = []
        for line in f:
            curr_sentence.append(line.strip())
            output_dict[line.strip()] = None
            if line[0] in ['.', '!', '?']:
                sentences.append(curr_sentence)
                curr_sentence = []
    return sentences, output_dict


def viterbi(sentence, init, trans, obs, tags=tags):
    """Run the Viterbi algorithm on the given sentence."""
    prob = [[0 for _ in range(len(tags))] for _ in range(len(sentence))]
    prev = [[None for _ in range(len(tags))] for _ in range(len(sentence))]
    for i in range(len(tags)):
        try:
            prob[0][i] = init[i] * obs[i][sentence[0]]
        except KeyError:
            prob[0][i] = 0  # TODO: come up with better unseen word handling strategy
        prev[0][i] = None  # unnecessary
    # Recursive case
    for t in range(1, len(sentence)):
        for i in range(len(tags)):
            val = float('-inf')
            max_tag = None
            for j in range(len(tags)):
                try:
                    curr_val = prob[t-1][j] * trans[i][j] * obs[i][sentence[t]]
                except KeyError:
                    curr_val = 0  # TODO: come up with better unseen word handling strategy
                if curr_val > val:
                    val = curr_val
                    max_tag = j
            prob[t][i] = val
            prev[t][i] = max_tag
    return prob, prev


def backtracking(sentence, prob, prev):
    tagged_words = []
    val = float('-inf')
    max_tag = None
    for i in range(len(tags)):
        if prob[-1][i] > val:
            val = prob[-1][i]
            max_tag = i
    tagged_words.append((sentence[-1], tags[max_tag]))
    for t in range(len(sentence) - 2, -1, -1):
        max_tag = prev[t + 1][max_tag]
        tagged_words.insert(0, (sentence[t], tags[max_tag]))
    return tagged_words


def output(output_file, training_list):
    """Output the tagged sentences to the output file."""
    initial_probs, observation_probs, transition_probs = train(training_list)
    sentences, output_dict = read_test_file(args.testfile)
    with open(output_file, 'w') as f:
        for sentence in sentences:
            prob, prev = viterbi(sentence, initial_probs, transition_probs, observation_probs)
            tagged_words = backtracking(sentence, prob, prev)
            for word, tag in tagged_words:
                f.write(f'{word} : {tag}\n')


def get_stats(output_file, test_file):
    """Get the accuracy of the output file."""
    with open(output_file, 'r') as f:
        with open(test_file, 'r') as g:
            correct = 0
            total = 0
            for line1, line2 in zip(f, g):
                if line1.strip() == line2.strip():
                    correct += 1
                total += 1
    return correct / total


def find_inaccuracies(output_file, test_file):
    """Find the words that were incorrectly tagged."""
    with open(output_file, 'r') as f:
        with open(test_file, 'r') as g:
            for line1, line2 in zip(f, g):
                if line1.strip() != line2.strip():
                    print(line1.strip(), line2.strip())




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
    # sentences = read_training_files(training_list)
    # o = get_observation_probs(sentences)
    # print(o)
    output(args.outputfile, training_list)
    # print("Accuracy: {}".format(get_stats(args.outputfile, 'training1.txt')))
    # find_inaccuracies(args.outputfile, 'training1.txt')