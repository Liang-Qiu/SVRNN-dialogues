"""Script for reading and processing the train/eval/test data
"""

import numpy as np
import pickle

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
DECODING_START = '<d>'
DECODING_END = '</d>'


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers).
    """
    def __init__(self, vocab_file, max_size, use_glove, glove_path):
        """Constructor.
        
        Args:
            vocab_file: string, path to the vocabulary file.
            max_size: int, maximum size of the vocabulary.
            use_glove: bool
            glove_path: string
        """
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_to_glove = {}
        self.id_to_glove = {}
        self.count = 0

        if use_glove:
            print('Use pre-train glove, loading the glove ...')
            word2glove = {}
            with open(glove_path, 'r', encoding='utf8') as glove_f:
                for line in glove_f:
                    line = line.split()
                    word = ''.join(line[:-300]).lower()
                    if word == 'pad':
                        word2glove[PAD_TOKEN] = np.asarray(line[-300:],
                                                           dtype='float')
                    elif word == 'sos':
                        word2glove[DECODING_START] = np.asarray(line[-300:],
                                                                dtype='float')
                    elif word == 'eos':
                        word2glove[DECODING_END] = np.asarray(line[-300:],
                                                              dtype='float')
                    elif word == 'unk':
                        word2glove[UNKNOWN_TOKEN] = np.asarray(line[-300:],
                                                               dtype='float')
                    else:
                        word2glove[word] = np.asarray(line[-300:],
                                                      dtype='float')

            # symbol = {UNKNOWN_TOKEN : 'unk', PAD_TOKEN : 'pad', DECODING_START : 'sos', DECODING_END : 'eos'}
            for w in [UNKNOWN_TOKEN, PAD_TOKEN, DECODING_START, DECODING_END]:
                self.word_to_id[w] = self.count
                self.id_to_word[self.count] = w
                self.word_to_glove[w] = word2glove[w]
                self.id_to_glove[self.count] = word2glove[w]
                self.count += 1

            with open(vocab_file, 'r') as vocab_f:
                for line in vocab_f:
                    pieces = line.split()
                    if len(pieces) != 2:
                        print(
                            'WARNING: incorrectly formatted line in vocabulary file: {}'
                            .format(line))
                        continue
                    if pieces[0] in self.word_to_id:
                        raise ValueError(
                            'Duplicated word in vocabulary file: {}.'.format(
                                pieces[0]))

                    if pieces[0] in word2glove:
                        self.word_to_id[pieces[0]] = self.count
                        self.id_to_word[self.count] = pieces[0]
                        self.word_to_glove[pieces[0]] = word2glove[pieces[0]]
                        self.id_to_glove[self.count] = word2glove[pieces[0]]
                        self.count += 1
                    if max_size != 0 and self.count >= max_size:
                        break

        else:
            for w in [UNKNOWN_TOKEN, PAD_TOKEN, DECODING_START, DECODING_END]:
                self.word_to_id[w] = self.count
                self.id_to_word[self.count] = w
                self.count += 1

            with open(vocab_file, 'r') as vocab_f:
                for line in vocab_f:
                    pieces = line.split()
                    if len(pieces) != 2:
                        print(
                            'WARNING: incorrectly formatted line in vocabulary file: {}'
                            .format(line))
                        continue
                    if pieces[0] in self.word_to_id:
                        raise ValueError(
                            'Duplicated word in vocabulary file: {}.'.format(
                                pieces[0]))
                    self.word_to_id[pieces[0]] = self.count
                    self.id_to_word[self.count] = pieces[0]
                    self.count += 1
                    if max_size != 0 and self.count >= max_size:
                        break
        print(
            'INFO: Finished reading {} of {} words in vocab, last word added: {}'
            .format(self.count, max_size, self.id_to_word[self.count - 1]))

        # with open('word.pkl', 'w') as f:
        #   pickle.dump(self.id_to_word, f)
        # assert 0

    def _word2id(self, word):
        """Returns the id (integer) of a word (string). Returns <UNK> id if word is OOV.
        """
        if word not in self.word_to_id:
            return self.word_to_id[UNKNOWN_TOKEN]
        return self.word_to_id[word]

    def _id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer).
        """
        if word_id not in self.id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id_to_word[word_id]

    def _word2glove(self, word):
        if word not in self.word_to_glove:
            return self.word_to_glove[UNKNOWN_TOKEN]
        return self.word_to_glove[word]

    def _id2glove(self, word_id):
        if word_id not in self.id_to_glove:
            raise ValueError('Id not found in glove: %d' % word_id)
        return self.id_to_glove[word_id]

    def _size(self):
        """Returns the total size of the vocabulary
        """
        return self.count


def article2ids(article_words, vocab: Vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.
    """
    ids = []
    oovs = []
    unk_id = vocab._word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab._word2id(w)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab._size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
  """
    ids = []
    unk_id = vocab._word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab._word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab._size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab):
    """Get words from output ids.
    """
    words = []
    for i in id_list:
        w = vocab._id2word(i)
        words.append(w)
    return words


def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs.
    """
    unk_id = vocab._word2id(UNKNOWN_TOKEN)
    words = article.split()
    vwords = []
    for w in words:
        if vocab._word2id(w) == unk_id:
            vwords.append("--%s--" % w)
        else:
            vwords.append(w)
    out_str = ' '.join(vwords)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    """Returns the abstract string, highlighting the OOVs.
    """
    unk_id = vocab._word2id(UNKNOWN_TOKEN)
    words = abstract.split()
    vwords = []
    for w in words:
        if vocab._word2id(w) == unk_id:
            if article_oovs is None:
                vwords.append("--%s--" % w)
            else:
                if w in article_oovs:
                    vwords.append("__%s__" % w)
                else:
                    vwords.append("--%s--" % w)
        else:
            vwords.append(w)
    out_str = ' '.join(vwords)
    return out_str
