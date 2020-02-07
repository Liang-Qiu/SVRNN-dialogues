import random
import torch
import torch.optim as optim
import numpy as np

from graph_matching.models.graph_structure import GSNModel
from graph_matching.datasets.vocab import Vocab
from graph_matching.datasets.UbuntuChatCorpus import Batcher
from graph_matching.config import get_default_config


def train(config, model, training_set, validation_set, optimizer):
    model.train()
    for i_iter in range(config['training']['n_training_steps']):
        batch = training_set._next_batch()
        if batch is None: break

        optimizer.zero_grad()
        vocab_dist = model(batch)
        print("Training ...")

        # loss.sum().backward()
        # optmizer.step()

        # if (i_iter + 1) % config['training']['print_after'] == 0:
        #     print('print metrics')


def main():
    config = get_default_config()

    # set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)

    # set device
    use_cuda = config['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # set data path
    config['data'][
        'data_path'] = config['data']['data_pre'] + 'train-sample/train-*.json'
    config['data']['eval_data_path'] = config['data'][
        'data_pre'] + 'dev-sample/dev-*.json'
    config['data']['test_data_path'] = config['data'][
        'data_pre'] + 'test-sample/test-*.json'
    config['data']['vocab_path'] = config['data']['data_pre'] + 'vocab'

    # vocabulary
    vocab = Vocab(config['data']['vocab_path'], config['data']['vocab_size'],
                  config['data']['use_glove'], config['data']['glove_path'])

    # build model
    model = GSNModel(config, vocab)

    # train or inference
    training_set = Batcher(config['data']['data_path'], vocab, config)
    validation_set = Batcher(config['data']['eval_data_path'], vocab, config)

    optimizer = optim.Adadelta(model.parameters(),
                               lr=config['training']['learning_rate'])

    # if hps.mode == 'train':
    #     batcher = Batcher(hps.data_path, vocab, hps)
    #     eval_hps = hps._replace(mode='eval')
    #     eval_batcher = Batcher(hps.eval_data_path, vocab, eval_hps)

    #     model = GSNModel(hps, vocab)
    #     train(model, batcher, eval_batcher, vocab, hps)
    # elif hps.mode == 'decode':
    #     decode_mdl_hps = hps._replace(max_dec_steps=1)
    #     batcher = Batcher(hps.test_data_path, vocab, decode_mdl_hps)  # for test

    #     model = GSNModel(decode_mdl_hps, vocab)
    #     decoder = BeamSearchDecoder(model, batcher, vocab)
    #     decoder._decode()

    train(config, model, training_set, validation_set, optimizer)

    if config['save_model']:
        torch.save(model.state_dict(), "graph_structure.pt")


if __name__ == "__main__":
    main()
