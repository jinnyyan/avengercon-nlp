import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re
from collections import Counter
from nltk import RegexpTokenizer

data_directory = "./training"
corpus_file = "./data/cleaned_corpus.txt"
ranked_vocab_file = "./data/results-brown.txt"

unk_num = 10  # Min number of frequency for word to not be replaced with 'UNK'


def main():
    pattern = r"[\w'.]+"

    # all words including "you've" is now a token
    vectorizer = CountVectorizer(token_pattern=pattern)

    # List of sentences with cleaned tokens
    all_data = load_data(data_directory)

    # Vectorizes data, identifies features and counts
    doc_term_matrix = vectorizer.fit_transform(all_data)
    vocab = vectorizer.get_feature_names()
    counts = doc_term_matrix.sum(axis=0).A1

    # Organizes data into organized dictionaries, creates UNK list and counts
    count_map = dict(zip(vocab, counts))
    unk_dict = dict((k, v) for k, v in count_map.items() if v <= unk_num)
    unk_list = unk_dict.keys()
    unk_amount = sum(unk_dict.values())

    # Completes the same, for non-UNK words and counts
    norm_dict = dict((k, v) for (k, v) in count_map.items() if v > unk_num)
    norm_dict['UNK'] = unk_amount
    norm_counter = Counter(norm_dict).most_common()

    # Finally, returns ranked vocab list and cleaned corpus
    if not os.path.exists("./data"):
        os.makedirs('./data')

    create_ranked_vocab_list(norm_counter)
    create_corpus(all_data, unk_list)


def create_ranked_vocab_list(word_dict):
    """
    word_dict : Dictionary of words in training corpus, in ranked order by freq
    * Creates ranked vocab list with '[word] [frequency]' format per line
    """
    with open(ranked_vocab_file, 'w+') as f:
        for i in word_dict:
            f.write(i[0]+' '+str(i[1]) + '\n')
    f.close()


def create_corpus(cdata, unk):
    """
    cdata : cleaned data returned from load_data(data)
    unk   : list of words to be replaced with UNK
    * Creates cleaned corpus file and replaces words with low freq
    * (on unknown list) with 'UNK'
    """
    count = 0
    with open(corpus_file, 'w+') as f:
        for sentence in cdata:
            count += 1
            print_progress("UNK replacement completion", count/len(cdata))
            for word in sentence.split():
                if word in unk:
                    sentence = re.sub(r'\b{}\b'.format(word), 'UNK', sentence)
            f.write(sentence+'\n')
    f.close()


def print_progress(heading, percent):
    """
    heading : Words to be printed at front of progress bar
    percent : Progress of completion
    * Displays arrow and percentage of completion of operation
    """
    bar_length = 20
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}%".format(
        heading, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def load_data(file_path):
    """
    file_path : Directory with all training data
    * Reads all files in directory, cleans words after tokenizing,
    * returns list of processed sentences.
    """
    dirs = os.listdir(file_path)
    all_data = list()
    for file in dirs:
        with open(file_path+'/'+file, 'r') as f:
            txt = f.readlines()
            tokenizer = RegexpTokenizer(r"\w[\w.']*/")
            # can include 'N.Y.' but not include '.'
        for line in txt:
            txt_tokens = tokenizer.tokenize(line)
            lowered_words = [t.lower().rstrip('/') for t in txt_tokens]
            new_txt = " ".join(lowered_words)
            all_data.append(new_txt)
    all_data = list(filter(None, all_data))
    return all_data


if __name__ == "__main__":
    main()
