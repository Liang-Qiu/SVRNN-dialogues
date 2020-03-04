"""Global parameters.
"""
word2vec_path = None  # The path to word2vec. Can be None.
data_dir = "data/data.pkl"  # Raw data directory.
work_dir = "working"  # Experiment results directory.
equal_batch = True  # Make each batch has similar length.
resume = False  # Resume from previous.
forward_only = False  # Only do decoding.
save_model = True  # Create checkpoints.
test_path = "run1500783422"  # The dir to load checkpoint for forward only.
result_path = "data/results.pkl"  # The dir to results.
use_test_batch = True  # Use test batch during testing.
n_state = 10  # Number of states.
api_dir = "data/cambridge_data/api_cambridge.pkl"  # "data/api_simdial_weather.pkl"#"data/cambridge_data/api_cambridge.pkl"
rev_vocab_dir = "data/cambridge_data/rev_vocab.pkl"  # "data/weather_rev_vocab.pkl"#"data/cambridge_data/rev_vocab.pkl"

temperature = 0.5  # temperature for gumbel softmax
use_cuda = False
seed = 233

# state variable
n_state = 10  # the number of states

# Network general
cell_type = "lstm"  # gru or lstm
encoding_cell_size = 400  # size of the rnn
state_cell_size = n_state
embed_size = 300  # word embedding size
max_utt_len = 40  # max number of words in an utterance
max_dialog_len = 10  # max number of turns in a dialog
num_layer = 1  # number of context RNN layers

# Optimization parameters
op = "adam"
grad_clip = 5.0  # gradient abs max cut
init_w = 0.08  # uniform random from [-init_w, init_w]
batch_size = 16  # mini-batch size
init_lr = 0.001  # initial learning rate
lr_hold = 1  # only used by SGD
lr_decay = 0.6  # only used by SGD
dropout = 0.4  # drop out rate
improve_threshold = 0.996  # for early stopping
patient_increase = 2.0  # for early stopping
early_stop = True
max_epoch = 60  # max number of epoch of training
grad_noise = 0.0  # inject gradient noise?

with_bow_loss = True
bow_loss_weight = 0.4  # weight of the bow_loss
n_epoch = 10
with_label_loss = False  # semi-supervised or not
with_BPR = True
with_direct_transition = False  # direct prior transition prob
with_word_weights = False

if with_word_weights:
    with open(rev_vocab_dir, "r") as fh:
        rev_vocab = pkl.load(fh)

    slot_value_id_list = []
    for k, v in rev_vocab.items():
        # print(type(k))
        if ("slot_" in k) or ("value_" in k):
            #print(k)
            slot_value_id_list.append(v)

    multiply_factor = 3
    one_weight = 1.0 / (len(rev_vocab) +
                        (multiply_factor - 1) * len(slot_value_id_list))
    word_weights = [one_weight] * len(rev_vocab)
    for i in slot_value_id_list:
        word_weights[i] = multiply_factor * word_weights[i]

    sum_word_weights = np.sum(word_weights)
    # print(sum_word_weights)
    # print(type(sum_word_weights))
    assert (sum_word_weights == float(1.0))
    word_weights = list(len(rev_vocab) * np.array(word_weights))

else:
    word_weights = None
