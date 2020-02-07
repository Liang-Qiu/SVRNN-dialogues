import re
import gzip
import io
import argparse
import numpy as np

from utils.io_utils import read_ubuntu_threads
from utils.tokenizer import tokenize_all

PUNCT = re.compile(r'[,:;]')


def get_dataset(corpus, fns, n_cands=2):
    """
    :param corpus: 1D: n_threads, 2D: n_utterances; elem=(time, speakerID, utterance)
    :return:
    """
    dataset = []
    dataset_fns = []
    n_samples = 0

    for fn, thread in zip(fns, corpus):
        conversation = []
        indices = range(len(thread))
        speaker_ids = get_speaker_ids(thread)

        # adr_indices: the index of the adr that is the same as one of the speaker ids
        thread, adr_ids, adr_indices = get_addressee_info(thread, speaker_ids)

        if len(adr_indices) < 10:
            continue

        for index, (line, adr_id) in enumerate(zip(
                thread, adr_ids)):  # sent: [time, name, tokens]
            pos_response = line[2]
            adr_label = '-'
            responses = [pos_response] + ['-' for i in range(n_cands - 1)]
            label = '-'

            if adr_id:
                neg_responses = get_neg_responses(thread, indices, index,
                                                  n_cands - 1)
                responses = [pos_response] + neg_responses

                np.random.shuffle(responses)
                label = responses.index(pos_response)
                adr_label = '<%s>' % adr_id

                n_samples += 1

            sample = [line[0], line[1]] + [adr_label] + responses + [label]
            conversation.append(sample)

        dataset.append(conversation)
        dataset_fns.append(fn)

    print('\tThreads: %d' % len(dataset))
    print('\tSamples: %d' % n_samples)

    return dataset, dataset_fns


def get_neg_responses(thread, indices, pos_sample_index, n_cands=1):
    neg_indices = list(set(indices) - set([pos_sample_index]))
    np.random.shuffle(neg_indices)
    return [thread[neg_indices[i]][2] for i in range(n_cands)]


def get_speaker_ids(sents):
    return list(set([sent[1][1:-1] for sent in sents]))


def get_addressee_info(thread, speaker_ids):
    adr_ids = []
    adr_indices = []

    for i, line in enumerate(thread):
        utterance = line[2]
        n_words = len(utterance)
        adr_id = utterance[0]

        if n_words > 2 and len(utterance[1]) == 1 and PUNCT.match(
                utterance[1]) and adr_id in speaker_ids:
            thread[i][2] = utterance[2:]
            adr_indices.append(i)
        else:
            adr_id = None

        adr_ids.append(adr_id)

    return thread, adr_ids, adr_indices


def tuning(data):
    return [[sent for sent in sents if len(sent[2]) > 0 and sent[1] != "*"]
            for sents in data]


def save(fn, dataset, fns):
    with gzip.open(fn + '.gz', 'wb') as gf:
        with io.TextIOWrapper(gf, encoding='utf-8') as enc:
            for f_name, thread in zip(fns, dataset):
                enc.writelines('# %s\n' % f_name)
                for line in thread:
                    text = '%s\t%s\t%s\t' % (line[0], line[1], line[2])

                    responses = [' '.join(r) for r in line[3:-1]]
                    for r in responses:
                        text += '%s\t' % r

                    text += '%s\n' % line[-1]

                    enc.writelines(text)
                enc.writelines('\n')


def main(argv):
    print('\nDATASET CREATION START')

    np.random.seed(0)

    # corpus: (threads, fn)
    # threads: 1D: n_threads, 2D: n_utterances; elem=(time, speakerID, utterance)
    corpus, fns = read_ubuntu_threads(argv.data, argv.en)

    tokenized_corpus = tokenize_all(corpus)
    tokenized_corpus = tuning(tokenized_corpus)

    dataset, fns = get_dataset(tokenized_corpus, fns, argv.n_cands)

    assert len(dataset) == len(fns), 'dataset: %d  fns: %d' % (len(dataset),
                                                               len(fns))
    fn = argv.data.split('/')[-1]
    save(fn, dataset, fns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Generator')

    # Data Creation
    parser.add_argument('--en', default=None, help='English files')
    parser.add_argument('--data',
                        default="/Users/liangqiu/Workspace/#ubuntu_test",
                        help='Data')

    parser.add_argument('--n_cands',
                        type=int,
                        default=2,
                        help='Num of candidates')

    argv = parser.parse_args()
    print
    print(argv)

    main(argv)
