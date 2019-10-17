import nltk
import itertools
import math
from collections import defaultdict, Counter
import pickle
import os

k_init = 200

corpus_file = "./data/cleaned_corpus.txt"
vocab_file = './data/results-brown.txt'
clusters_results_file = './data/brownclusters.txt'

bg_N = 0 # Used for quality equation, N value of bigrams
unig_N = 0 # Used for quality equation, N value of unigrams

bigram_word_counts = defaultdict(lambda:0)
unigram_word_counts = defaultdict(lambda:0)
ordered_vocab = list()

binary_dictionary = defaultdict(lambda: '')


def main():
    global bg_N, unig_N, bigram_word_counts, unigram_word_counts, ordered_vocab
    
    unigram_word_counts = load_ranked_vocab()
    del unigram_word_counts['UNK']
    corpus = load_corpus()

    # for small sample---------------------------------------------------------------------
    # unigram_word_counts = {'the':6, 'house':4, 'is': 4, 'by':4, 'dog':2,'cat':2, 'that':2}
    # corpus = ['the dog is by the house',
    #           'the cat is by the house',
    #           'the dog is by that house',
    #           'the cat is by that house']
    # --------------------------------------------------------------------------------------

    print("Done loading vocab and corpus from file.")

    bigram_word_counts = get_bigrams(corpus)
    ordered_vocab = list(unigram_word_counts.keys())
    print("Done retrieving initial bigram counts and ordered vocab list (by decreasing frequency).")

    bg_N = sum([sum(x.values()) for x in bigram_word_counts.values()]) #658953
    unig_N = sum(unigram_word_counts.values()) #699397, 623718 without UNK

    run_brown()

    if not os.path.exists("./data"):
        os.makedirs('./data') 

    with open(clusters_results_file, 'w') as f:
        for word in binary_dictionary:
            f.write(word + '\t\t\t\t\t' + binary_dictionary[word] + '\n')
    f.close()


def load_corpus():
    """
    * Loads corpus file into data structure
    """
    with open(corpus_file, 'r') as f:
        corpus = f.readlines()
        return corpus

def load_ranked_vocab():
    """
    * Loads ranked vocab file into data structure
    """
    with open(vocab_file, 'r') as f:
        txt = f.readlines()
        data = [x.split() for x in txt]
        n = {x[0]:int(x[1]) for x in data}
        return n

def get_bigrams(corpus):
    """
    corpus : Clean corpus taken from file
    * Generates all bigrams from corpus (dictionary of dictionary)
    """
    bigram_model = defaultdict(lambda : defaultdict(lambda :0))
    for line in corpus:
        tokens = line.split()
        bgs = nltk.bigrams(tokens)
        fdist = nltk.FreqDist(bgs)
        for k,v in fdist.items():
            bigram_model[k[0]][k[1]] += v
    return bigram_model

def run_brown():
    global unigram_word_counts, bigram_word_counts
    cluster_to_word = list() # ordered list where index is cluster # to word
    word_to_cluster = dict()

    curr_unigram_counts = list()
    curr_bigram_counts = list()

    ################ ONLY DONE ONCE ##########################
    for i in range(k_init): # initialize C to be the top 200 words first
        curr_word = ordered_vocab.pop(0)

        word_to_cluster[curr_word] = i
        cluster_to_word.append([curr_word])
        curr_unigram_counts.append(unigram_word_counts[curr_word])
        curr_bigram_counts.append(bigram_word_counts[curr_word])

    for i in range(k_init):
        for j in list(curr_bigram_counts[i]):
            if j in word_to_cluster:
                curr_bigram_counts[i][word_to_cluster[j]] = curr_bigram_counts[i].pop(j)
            else:
                del curr_bigram_counts[i][j]

    ##########################################################
    for next_word in ordered_vocab:
        print("Adding word '" + next_word + "' into the mix")
        cluster_to_word.append([next_word])
        word_to_cluster[next_word] = k_init # new cluster (quality computer) after adding new word

        curr_unigram_counts.append(unigram_word_counts[next_word])
        next_bigram_word_counts = bigram_word_counts[next_word]

        for i in list(next_bigram_word_counts):
            if i in word_to_cluster:
                next_bigram_word_counts[word_to_cluster[i]] = next_bigram_word_counts.pop(i)
            else:
                del next_bigram_word_counts[i]

        curr_bigram_counts.append(next_bigram_word_counts)

        max_quality = 0
        arg_max = (0,0)

        #startingvals, starting_quality = get_quality(curr_unigram_counts, curr_bigram_counts)

        for first_i in list(range(k_init)):
            for second_i in list(range(first_i+1, k_init+1)):
                temp_unigram = curr_unigram_counts.copy()
                temp_bigram = curr_bigram_counts.copy()
                temp_cluster_to_word = cluster_to_word.copy()
                current_merge = (first_i, second_i)

                a = temp_unigram.pop(second_i)
                b = temp_unigram.pop(first_i)
                temp_unigram.insert(first_i, a+b)

                x = Counter(temp_bigram.pop(second_i))
                y = Counter(temp_bigram.pop(first_i))
                combined = dict(x+y)

                if second_i in combined:
                    if first_i in combined:
                        combined[first_i] = combined[first_i] + combined.pop(second_i)
                    else:
                        combined[first_i] = combined.pop(second_i)

                new_combined = dict()
                for x in list(combined):
                    if x > second_i:
                        new_combined[x - 1] = combined.pop(x)
                    else:
                        new_combined[x] = combined.pop(x)

                temp_bigram.insert(first_i, new_combined)

                # temp_word_to_cluster = {key:first_i if (word_to_cluster[key]==second_i or word_to_cluster[key]==first_i) else
                #                     word_to_cluster[key] for key in word_to_cluster.keys()} # make both indices the first index
                # temp_word_to_cluster = {key:(temp_word_to_cluster[key] - 1) if (temp_word_to_cluster[key] > second_i) else
                #                         temp_word_to_cluster[key] for key in temp_word_to_cluster.keys()} # lower all the rest of the indices after second index

                temp_cluster_to_word[first_i] = temp_cluster_to_word[first_i] + temp_cluster_to_word.pop(second_i)

                curr_quality = get_quality(temp_unigram, temp_bigram)

                if curr_quality > max_quality:
                    max_quality = curr_quality
                    arg_max = current_merge

                    best_cluster_to_word = temp_cluster_to_word
                    # best_word_to_cluster = temp_word_to_cluster
                    best_unigram_counts = temp_unigram
                    best_bigram_counts = temp_bigram
                    print("- - -  Quality replaced by: " + str(arg_max) + " with value " + str(max_quality))


        add_to_binary(arg_max, cluster_to_word) # This actually is the old cluster to word

        curr_unigram_counts = best_unigram_counts
        curr_bigram_counts = best_bigram_counts
        # word_to_cluster = best_word_to_cluster
        cluster_to_word = best_cluster_to_word

        print("Merged {0} and {1} together".format(arg_max[0], arg_max[1]))



    # pickle.dump(curr_unigram_counts, open('curr_unigram_counts.pkl', 'wb'))
    # pickle.dump(curr_bigram_counts, open('curr_bigram_counts.pkl', 'wb'))
    # pickle.dump(word_to_cluster, open('word_to_cluster.pkl', 'wb'))
    # pickle.dump(cluster_to_word, open('cluster_to_word.pkl', 'wb'))

    print("Done with merging to k_init clusters")

    # for i in range(k_init)[::-1]: # reverse, backward descending
    #     print("Running final merging "+str(i))
    #
    #     max_quality = 0
    #     arg_max = (0,0)
    #
    #     for first_i in range(i):
    #         for second_i in range(first_i+1, i):
    #             print(first_i, second_i)
    #             temp_unigram = curr_unigram_counts.copy()
    #             temp_bigram = curr_bigram_counts.copy()
    #             temp_cluster_to_word = cluster_to_word.copy()
    #
    #             current_merge = (first_i, second_i)
    #             a = temp_unigram.pop(second_i)
    #             b = temp_unigram.pop(first_i)
    #             temp_unigram.insert(first_i, a+b)
    #
    #             x = Counter(temp_bigram.pop(second_i))
    #             y = Counter(temp_bigram.pop(first_i))
    #             combined = dict(x + y)
    #
    #             temp_bigram.insert(first_i, combined)
    #
    #             for ind in range(len(temp_bigram)):
    #                 part = temp_bigram[ind]
    #                 if second_i in part:
    #                     if first_i in part:
    #                         part[first_i] = part[first_i] + part[second_i]
    #                     else:
    #                         part[first_i] = part[second_i]
    #
    #                 new_part = dict()
    #                 for x in list(part):
    #                     if x > second_i:
    #                         new_part[x - 1] = part.pop(x)
    #                     else:
    #                         new_part[x] = part.pop(x)
    #
    #                 del part
    #                 temp_bigram.insert(ind, new_part)
    #
    #
    #
    #
    #             # temp_word_to_cluster = {key:first_i if (word_to_cluster[key]==second_i or word_to_cluster[key]==first_i) else
    #             #                     word_to_cluster[key] for key in word_to_cluster.keys()} # make both indices the first index
    #             # temp_word_to_cluster = {key:(temp_word_to_cluster[key] - 1) if (temp_word_to_cluster[key] > second_i) else
    #             #                         temp_word_to_cluster[key] for key in temp_word_to_cluster.keys()} # lower all the rest of the indices after second index
    #
    #             temp_cluster_to_word[first_i]= temp_cluster_to_word[first_i] + temp_cluster_to_word.pop(second_i)
    #
    #             curr_quality = get_quality(temp_unigram, temp_bigram)# temp_word_to_cluster)
    #             if curr_quality > max_quality:
    #                 max_quality = curr_quality
    #                 arg_max = current_merge
    #
    #                 best_cluster_to_word = temp_cluster_to_word
    #                 # best_word_to_cluster = temp_word_to_cluster
    #                 best_unigram_counts = temp_unigram
    #                 best_bigram_counts = temp_bigram
    #                 print("- - -  Quality replaced by: " + str(arg_max) + " with value " + str(max_quality))
    #
    #     add_to_binary(arg_max, cluster_to_word)
    #
    #     curr_unigram_counts = best_unigram_counts
    #     curr_bigram_counts = best_bigram_counts
    #     # word_to_cluster = best_word_to_cluster
    #     cluster_to_word = best_cluster_to_word
    #
    #     print("Merged {0} and {1} together".format(arg_max[0], arg_max[1]))



def add_to_binary(arg_max_tuple, cluster_to_word):
    global binary_dictionary

    first = arg_max_tuple[0]
    second = arg_max_tuple[1]

    for word in cluster_to_word[first]:
        print("Adding 0 to "+word)
        binary_dictionary[word] = '0' + binary_dictionary[word]

    for word in cluster_to_word[second]:
        print("Adding 1 to "+word)
        binary_dictionary[word] = '1' + binary_dictionary[word]



def get_quality(unigram, bigram):
    quality = 0
    #vallist = list()
    for first_cluster in range(len(bigram)):
        for second_cluster in bigram[first_cluster]:
            part1 = bigram[first_cluster][second_cluster] / bg_N
            part2 = unigram[first_cluster] / unig_N
            try:
                part3 = unigram[second_cluster - 1] / unig_N
            except IndexError:
                part3 = unigram[second_cluster] / unig_N
            val = part1 * math.log(part1 / (part2 * part3))

            # vallist.append(val)
            quality += val
    return abs(quality) # TODO: Not sure this is ok

#
# def get_change_quality(vallist, startquality, merge, unigram, bigram, word_to_cluster):
#
#     first_cluster = merge[0]
#     second_cluster = merge[1]
#
#     for i in range(len(vallist)):
#         if first_cluster in bigram[i]:
#             if second_cluster in bigram[i]:
#                 bigram[i][first_cluster]bigram[i].pop(second_cluster)
#         new_i = i -
#
#     a = vallist.pop(first_cluster)
#     b = vallist.pop(second_cluster)
#
#
#
#     vallist.insert(first_cluster, a + b)
#

    #     second_clusters = {word_to_cluster[word]: second_words[word] for word in second_words.keys() if
    #                        word in word_to_cluster}
    #
    #     for second_cluster in second_clusters:
    #         part1 = second_clusters[second_cluster] / bg_N
    #         part2 = unigram[first_cluster] / unig_N
    #         part3 = unigram[second_cluster] / unig_N
    #         val = part1 * math.log(part1 / (part2 * part3))
    #
    #         vallist.append(val)
    #         quality += val
    # return vallist, abs(quality)  # TODO: Not sure this is ok




# def find_best_merge(cluster_to_word):
#     global unigram_word_counts, bigram_word_counts
#     max_quality = 0
#     best_temp_c_to_w = list()
#     arg_max = (0,0)
#
#     for first_i in list(range(k_init)):
#         for second_i in list(range(first_i+1, k_init+1)):
#             temp_c_to_w = cluster_to_word.copy()
#             first_word = temp_c_to_w[first_i]
#             second_word = temp_c_to_w.pop(second_i)
#             temp_c_to_w[first_i] = (first_word, second_word)
#             print(temp_c_to_w)
#             temp_quality, temp_bigram_counts, temp_unigram_counts  = get_quality(temp_c_to_w)
#
#             if temp_quality > max_quality:
#                 print("Replaced max quality {0} with {1}".format(max_quality, temp_quality))
#                 max_quality = temp_quality
#
#                 best_temp_c_to_w = temp_c_to_w
#                 best_temp_c_to_w[first_i] = first_word # first_i? TODO
#
#                 arg_max = (first_word, second_word)
#     unigram_word_counts[first_word] += unigram_word_counts[second_word] # TODO
#     # for x in ordered_vocab: ###############need to include this too
#     #     if bigram_word_counts[(second_word, x)] > 1:
#     #         bigram_word_counts[(first_word, x)] += bigram_word_counts[(second_word, x)]
#     return arg_max, best_temp_c_to_w
#
#



# def get_temp_bigram_counts(cluster_to_word):
#     """
#     :param cluster_to_word: A list of where index is cluster # and contents are the words in it (word or tuple)
#     :return: Using the current words introduced, returns bigram counts in the form of (cluster, cluster) -> count
#     """
#     temp_bigram_counts = defaultdict(lambda: defaultdict(lambda:0))
#     for first_i in list(range(k_init)):
#         for second_i in list(range(first_i + 1, k_init)):
#             curr_w = cluster_to_word[first_i]
#             next_w = cluster_to_word[second_i]
#             if type(curr_w) == tuple:
#                 check1 = bigram_word_counts[curr_w[0]]
#                 check2 = bigram_word_counts[curr_w[1]]
#                 if next_w in check1:
#                     temp_bigram_counts[first_i][second_i] += check1[next_w]
#                 if next_w in check2:
#                     temp_bigram_counts[first_i][second_i] += check2[next_w]
#
#                 temp_bigram_counts[first_i][second_i] +=  + bigram_word_counts[curr_w[1]][next_w]
#                 temp_bigram_counts[second_i][first_i] += bigram_word_counts[next_w][curr_w[0]] + bigram_word_counts[next_w][curr_w[1]]
#             elif type(next_w) == tuple:
#                 check1 = (curr_w, next_w[0])
#                 check2 = (curr_w, next_w[1])
#                 check3 = (next_w[0], curr_w)
#                 check4 = (next_w[1], curr_w)
#                 if check1 in bigram_word_counts:
#                     temp_bigram_counts[(first_i, second_i)] = bigram_word_counts[check1]
#                 if check2 in bigram_word_counts:
#                     temp_bigram_counts[(first_i, second_i)] += bigram_word_counts[check2]
#                 if check3 in bigram_word_counts:
#                     temp_bigram_counts[(second_i, first_i)] += bigram_word_counts[check3]
#                 if check4 in bigram_word_counts:
#                     temp_bigram_counts[(second_i, first_i)] += bigram_word_counts[check4]
#             else: # if neither are tuples
#                 check1 = (curr_w, next_w)
#                 check2 = (next_w, curr_w)
#                 if check1 in bigram_word_counts:
#                     temp_bigram_counts[(first_i, second_i)] = bigram_word_counts[check1]
#                 if check2 in bigram_word_counts:
#                     temp_bigram_counts[(second_i, first_i)] += bigram_word_counts[check2]
#     return temp_bigram_counts
#
#
# def get_temp_unigram_counts(cluster_to_word):
#     """
#     :param cluster_to_word: A list of where index is cluster # and contents are the words in it (word or tuple)
#     :return: Using the current words introduced, returns unigram counts in the form of cluster -> count
#     """
#     temp_unigram_counts = dict()
#     for i in range(len(cluster_to_word)):
#         curr_w = cluster_to_word[i]
#         if type(curr_w) != tuple: # if this entry is only one word
#             temp_unigram_counts[i] = unigram_word_counts[curr_w]
#         else:  # if this entry is more than one word
#             temp_unigram_counts[i] = unigram_word_counts[curr_w[0]] + unigram_word_counts[curr_w[1]]
#     return temp_unigram_counts


if __name__=="__main__":
    main()

