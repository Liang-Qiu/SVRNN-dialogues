import torch


def get_default_config():
    """The default configs."""
    use_cuda = torch.cuda.is_available()
    save_model = False
    # node_state_dim = 32  # number of features for a node
    # graph_rep_dim = 128  # number of features of a graph representation
    # graph_embedding_net_config = dict(
    #     node_state_dim=node_state_dim,
    #     edge_feat_dim=1,
    #     edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],  # sizes of the hidden layers of the edge message nets.
    #     node_hidden_sizes=[node_state_dim * 2],  # sizes of the hidden layers of the node update nets.
    #     n_prop_layers=5,  # number of graph propagation layers.
    #     # set to False to not share parameters across message passing layers
    #     share_prop_params=True,
    #     # initialize message MLP with small parameter weights to prevent
    #     # aggregated message vectors blowing up, alternatively we could also use
    #     # e.g. layer normalization to keep the scale of these under control.
    #     edge_net_init_scale=0.1,
    #     # other types of update like `mlp` and `residual` can also be used here.
    #     node_update_type='gru',
    #     # set to False if your graph already contains edges in both directions.
    #     use_reverse_direction=True,
    #     # set to True if your graph is directed
    #     reverse_dir_param_different=False,
    #     # we didn't use layer norm in our experiments but sometimes this can help.
    #     layer_norm=False)
    # graph_matching_net_config = graph_embedding_net_config.copy()
    # graph_matching_net_config['similarity'] = 'dotproduct'
    # config for ubuntu dialogue corpus
    ubuntu_dialog_corpus_config = dict(
        data_size=345000,
        vocab_size=51165,
        data_pre="/home/liang/Workspace/#ubuntu_test/",
        data_path=None,
        eval_data_path=None,
        test_data_path=None,
        vocab_path=None,
        use_glove=False,
        glove_path="/home/liang/Workspace/glove.840B.300d.txt",
    )
    graph_structure_config = dict(
        branch_batch_size=100,
        sen_batch_size=9,
        emb_dim=300,
        sen_hidden_dim=300,
        branch_hidden_dim=300,
        max_enc_steps=50,
        max_dec_steps=50,
        min_dec_steps=5,
        dropout=1.0,
        positional_enc=True,  # use the positional encoding tricks
        positional_enc_dim=64,  # dimension of word embeddings
        n_gram=3,  # number of n_gram
        use_norm=True,  # use norm
        norm_alpha=0.25,  # norm_alpha
        user_struct=True,  # use the struct of user relation
        long_attn=False,  # use the struct of all sent attn
    )

    return dict(
        # encoder=dict(node_feat_dim=1, # GraphEditDistance task only cares about graph structure.
        #              edge_feat_dim=1,
        #              node_hidden_sizes=[node_state_dim],
        #              edge_hidden_sizes=None),
        # aggregator=dict(node_input_size=node_state_dim,
        #                 node_hidden_sizes=[graph_rep_dim],
        #                 graph_transform_sizes=[graph_rep_dim],
        #                 gated=True,
        #                 aggregation_type='sum'),
        # graph_embedding_net=graph_embedding_net_config,
        # graph_matching_net=graph_matching_net_config,
        # # Set to `embedding` to use the graph embedding net.
        # model_type='matching',
        graph_structure_net=graph_structure_config,
        # data=dict(
        #     problem='graph_edit_distance',
        #     dataset_params=dict(
        #         # always generate graphs with 20 nodes and p_edge=0.2.
        #         n_nodes_range=[20, 20],
        #         p_edge_range=[0.2, 0.2],
        #         n_changes_positive=1,
        #         n_changes_negative=2,
        #         validation_dataset_size=1000)),
        data=ubuntu_dialog_corpus_config,
        training=dict(
            batch_size=20,
            learning_rate=1e-3,
            mode='pair',
            loss='margin',
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=100,
            # Evaluate on validation set every `eval_after * print_after` steps.
            eval_after=10),
        evaluation=dict(batch_size=20),
        mode="train",
        seed=8,
        use_cuda=use_cuda,
        save_model=save_model,
    )
