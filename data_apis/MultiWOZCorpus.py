from __future__ import print_function
from __future__ import division

import json
from collections import Counter
import numpy as np
import nltk
import gensim
from loguru import logger

import params


class MultiWOZCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self,
                 corpus_path,
                 max_vocab_cnt=10000,
                 word2vec=None,
                 word2vec_dim=300,
                 labeled=False):
        """
        :param corpus_path: the folder that contains the MultiWOZ dialog corpus
        """
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.dialog_id = 0
        self.utt_id = 1
        self.label_id = 2
        self.labeled = labeled
        self.sil_utt = ["<s>", "<sil>", "</s>"]

        data = json.load(open(self._path, "r"))
        data = data[params.test_domain]
        self.train_corpus = self.process(data, labeled=self.labeled)
        logger.info(
            f"Number of dialogs in train set: {len(self.train_corpus[self.dialog_id])}"
        )
        logger.info("Printing dialog examples from train set")
        for i in range(3):
            dialog = self.train_corpus[self.dialog_id][i]
            logger.info("Example %d: %s" % (i, dialog))

        self.build_vocab(max_vocab_cnt)
        self.load_word2vec()
        logger.info("Done loading corpus")

    def process(self, data, labeled=False):
        new_dialog = []
        if labeled:
            new_labels = []
        new_utts = []
        all_lenes = []

        for dial in data:
            dialog = []
            dial_text = dial["text"]
            for turn in dial_text:
                usr_utt = ["<s>"] + nltk.WordPunctTokenizer().tokenize(
                    turn.split(' | ')[0].lower()) + ["</s>"]
                sys_utt = ["<s>"] + nltk.WordPunctTokenizer().tokenize(
                    turn.split(' | ')[1].lower()) + ["</s>"]
                new_utts.append(usr_utt)
                new_utts.append(sys_utt)

                all_lenes.extend([len(usr_utt)])
                all_lenes.extend([len(sys_utt)])

                dialog.append([usr_utt, sys_utt])
            new_dialog.append(dialog)
            if labeled:
                new_labels.append(dial["label"])

        logger.info("Max utt len %d, mean utt len %.2f" %
                    (np.max(all_lenes), float(np.mean(all_lenes))))

        if labeled:
            return new_dialog, new_utts, new_labels
        else:
            return new_dialog, new_utts

    def build_vocab(self, max_vocab_cnt):
        all_words = []
        for tokens in self.train_corpus[self.utt_id]:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        logger.info(
            "Load corpus with train size {}, \n raw vocab size {}, vocab size {} "
            "at cut_off {} OOV rate {}".format(
                len(self.train_corpus[0]), raw_vocab_size, len(vocab_count),
                vocab_count[-1][1],
                float(discard_wc) / len(all_words)))

        self.vocab = ["<pad>", "<unk>"] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab["<unk>"]
        self.id_to_vocab = {self.rev_vocab[v]: v for v in self.rev_vocab}

    def load_word2vec(self, binary=True):
        if self.word_vec_path is None:
            return
        raw_word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            self.word_vec_path, binary=binary)
        print("load w2v done")
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            if v not in raw_word2vec:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = raw_word2vec[v]
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" %
              (float(oov_cnt) / len(self.vocab)))

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                # convert utterance and feature into numeric numbers
                for usr_sent, sys_sent in dialog:
                    temp_turn = [[
                        self.rev_vocab.get(t, self.unk_id) for t in usr_sent
                    ], [self.rev_vocab.get(t, self.unk_id) for t in sys_sent]]
                    temp.append(temp_turn)
                results.append(temp)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])

        for i in range(3):
            logger.info(f"Example ID %d: %s" % (i, id_train[i]))

        return {'train': id_train}

    def get_labels(self):
        def _to_label_corpus(data):
            results = []
            for label in data:
                padded_label = label
                for _ in range(params.max_dialog_len - len(label)):  # padding
                    padded_label.append(-1)
                results.append(padded_label[:params.max_dialog_len])
            return results

        if self.labeled:
            id_labeled = _to_label_corpus(self.train_corpus[self.label_id])

        return {'labels': id_labeled}
