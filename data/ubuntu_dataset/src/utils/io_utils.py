import sys
import os
import io
import re
import gzip
import pickle as cPickle
import glob
import json

import numpy as np


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def read_ubuntu_threads(dir_name, lang_scores=None):
    ######################################################################################################
    # An example of lang scores:                                                                         #
    # ../data/2015-01/2015-01-02/2015-01-02-ubuntu-cn.txt:[et:0.7142841673745188, en:0.2857135229031523] #
    ######################################################################################################

    ########################################
    # Pick up the files written in English #
    ########################################
    en_file_names = []
    if lang_scores:
        en_file_names = get_en_file_names(lang_scores)

    os.chdir(dir_name)
    fns = glob.glob('*.txt')

    if lang_scores:
        fns = list(set(fns).intersection(set(en_file_names)))

    print(fns)

    regexp = re.compile(r'^[\x20-\x7E]+$')
    threads = []
    thread_fns = []
    for fn in fns:
        thread = []
        with open(fn) as fin:
            for line in fin.readlines():
                line = line.rstrip()
                if line.startswith('[') and regexp.search(line) is not None:
                    text = line.split()
                    thread.append([text[0], text[1], " ".join(text[2:])])

        if thread:
            threads.append(thread)
            thread_fns.append(fn)

    return threads, thread_fns


def get_en_file_names(fn, threshold=0.99):
    en_file_names = []

    for line in file(fn).readlines():
        line = line.rstrip()

        if len(line) == 0:
            continue

        index = line.index(':')
        file_path = line[:index].split('/')
        probs = [p.strip().split(':') for p in line[index + 2:-1].split(',')]

        lang = probs[0][0]
        prob = float(probs[0][1])

        if lang == 'en' and prob > threshold:
            en_file_names.append(file_path[-1])

    return en_file_names


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)


def load_data_sample(fn):
    """
    :param fn:
    :return: samples: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response, ... , label)
    """
    samples = []
    sample = []
    with gzip.open(fn, 'rb') as gf:
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 6:
                samples.append(sample)
                sample = []
            else:
                sample.append(line)
    return samples


def separate_dataset(fn):
    samples = []
    sample = []
    with gzip.open(fn, 'rb') as gf:
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 3:
                samples.append(sample)
                sample = []
            else:
                sample.append(line)

    cand_indices = range(len(samples))
    np.random.shuffle(cand_indices)
    dataset = [samples[i] for i in cand_indices]

    folded = len(dataset) / 20
    train_data = dataset[:folded * 17]
    dev_data = dataset[folded * 17:folded * 18]
    test_data = dataset[folded * 18:]

    save('train-data', train_data)
    save('dev-data', dev_data)
    save('test-data', test_data)
    print('Train: %d\tDev: %d\tTest: %d' %
          (len(train_data), len(dev_data), len(test_data)))
    return samples


def count_dataset(fn):
    samples = []
    sample = []
    num_w = 0
    num_s = 0
    with gzip.open(fn, 'rb') as gf:
        for line in gf:
            line = line.rstrip().split("\t")
            if len(line) < 3:
                samples.append(sample)
                sample = []
            else:
                sample.append(line)
                num_w += len(line[2])
                num_s += 1

    print('Total Words: %d\tWords per Sample: %f\tSamples: %d' %
          (num_w, num_w / float(num_s), num_s))


def load_dataset(fn, vocab=dict(), sample_size=1000000, check=False):
    """
    :return: samples: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    samples = []
    sample = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn) as gf:
        with io.TextIOWrapper(gf, encoding='utf-8') as dec:
            # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
            for line in dec:
                line = line.rstrip().split("\t")
                if len(line) < 6:
                    samples.append(sample)
                    sample = []

                    if len(samples) >= sample_size:
                        break
                else:
                    for i, sent in enumerate(line[3:-1]):
                        word_ids = []
                        for w in sent.split():
                            w = w.lower()
                            if w in vocab:
                                vocab[w] += 1
                            else:
                                vocab[w] = 1
                            word_ids.append(w)
                        line[3 + i] = word_ids

                    ################################
                    # Label                        #
                    # -1: Not related utterances   #
                    # 0-: Candidate response index #
                    ################################
                    line[-1] = -1 if line[-1] == '-' else int(line[-1])
                    sample.append(line)
    if check:
        say('\n\n LOAD DATA EXAMPLE:\n\t%s' % str(samples[0][0]))

    vocab_sorted = {
        k: v
        for k, v in sorted(
            vocab.items(), key=lambda item: item[1], reverse=True)
    }
    return samples, vocab_sorted


def output_samples(fn, samples, vocab_word=None):
    if samples is None:
        return

    if not os.path.exists(fn):
        os.mkdir(fn)

    for i, sample in enumerate(samples):
        with open(os.path.join(fn, fn + str(i) + '.json'), "w") as json_file:
            sample_dict = {}
            for i, s in enumerate(sample.context):
                sample.context[i] = " ".join(s)

            sample_dict['context'] = sample.context
            sample_dict['answer'] = " ".join(sample.answer)
            sample_dict['ans_idx'] = sample.ans_idx
            sample_dict['relation_at'] = sample.relation_at
            sample_dict['relation_user'] = sample.relation_user
            json.dump(sample_dict, json_file)


def output_vocab(word_vocab):
    with open('vocab', "w") as f:
        for i, word in enumerate(word_vocab):
            f.write(word + " " + str(i) + "\n")


def get_word(w, vocab_word=None):
    if vocab_word:
        return vocab_word.get_word(w)
    return w


def save(fn, data):
    with gzip.open(fn + '.gz', 'wb') as gf:
        for f in data:
            for line in f:
                text = '%s\t%s\t%s\n' % (line[0], line[1], line[2])
                gf.writelines(text)
            gf.writelines('\n')
