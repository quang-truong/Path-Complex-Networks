from models.complex_models.molecular_models import EmbedSparseCIN, OGBEmbedSparseCIN
from models.complex_models.tu_models import SparseCIN
from models.graph_models.gin_models import GIN0
from lib.utils.log_utils import str_arg_to_bool_list

def get_complex_model(args, dataset, device):
    if args.readout_dims is not None:
        args.readout_dims = tuple(sorted(args.readout_dims))
    else:
        args.readout_dims = tuple(range(args.max_dim+1))
    use_coboundaries = True if args.use_coboundaries == 'True' else False
    use_boundaries = True if args.use_boundaries == 'True' else False
    disable_graph_norm = True if args.disable_graph_norm == 'True' else False
    # Instantiate model
    # NB: here we assume to have the same number of features per dim
    if args.model == 'embed_sparse_cin':
        model = EmbedSparseCIN(dataset.num_node_type,                       # The number of atomic types
                            dataset.num_edge_type,                          # The number of bond types
                            dataset.num_classes,                            # num_classes
                            args.num_layers,                                # num_layers
                            args.emb_dim,                                   # hidden
                            dropout_rate=args.drop_rate,                    # dropout rate
                            in_dropout_rate=args.indrop_rate,               # dropout rate for input
                            max_dim=dataset.max_dim,                        # max_dim
                            jump_mode=args.jump_mode,                       # jumping knowledge
                            nonlinearity=args.nonlinearity,                 # nonlinearity
                            readout=args.readout,                           # readout
                            final_readout=args.final_readout,               # final readout
                            apply_dropout_before=args.drop_position,        # where to apply dropout
                            use_coboundaries=use_coboundaries,              # whether to use coboundaries
                            use_boundaries=use_boundaries,
                            include_down_adj=args.include_down_adj,
                            init_reduce=args.init_method,
                            embed_edge=args.use_edge_features,              # whether to use edge feats
                            graph_norm=args.graph_norm,                     # normalization layer
                            disable_graph_norm=disable_graph_norm,
                            readout_dims=args.readout_dims,                 # readout_dims
                            num_layers_update=args.num_layers_update,       # number of update layers
                            num_layers_combine=args.num_layers_combine,     # number of combine layers
                            num_fc_layers=args.num_fc_layers,               # number of fc layers after msg prop
                            conv_type=args.conv_type,                       # conv type A or B
                            ).to(device)
    elif args.model == 'ogb_embed_sparse_cin':
        model = OGBEmbedSparseCIN(dataset.num_tasks,                            # out_size
                                  args.num_layers,                              # num_layers
                                  args.emb_dim,                                 # hidden
                                  dropout_rate=args.drop_rate,                  # dropout_rate
                                  in_dropout_rate=args.indrop_rate,             # in-dropout_rate
                                  max_dim=dataset.max_dim,                      # max_dim
                                  jump_mode=args.jump_mode,                     # jumping knowledge
                                  nonlinearity=args.nonlinearity,               # nonlinearity
                                  readout=args.readout,                         # readout
                                  final_readout=args.final_readout,             # final readout
                                  apply_dropout_before=args.drop_position,      # where to apply dropout
                                  use_coboundaries=use_coboundaries,            # whether to use coboundaries
                                  use_boundaries=use_boundaries,
                                  include_down_adj=args.include_down_adj,
                                  init_reduce=args.init_method,
                                  embed_edge=args.use_edge_features,            # whether to use edge feats
                                  graph_norm=args.graph_norm,                   # normalization layer
                                  disable_graph_norm=disable_graph_norm,
                                  readout_dims=args.readout_dims,               # readout_dims
                                  num_layers_update=args.num_layers_update,     # number of update layers
                                  num_layers_combine=args.num_layers_combine,   # number of combine layers
                                  num_fc_layers=args.num_fc_layers,             # number of fc layers after msg prop
                                  conv_type=args.conv_type,                     # conv type A or B
                                  ).to(device)
    elif args.model == 'sparse_cin':
        model = SparseCIN(dataset.num_features_in_dim(0),             # num_input_features
                        dataset.num_node_labels,
                        dataset.num_classes,                          # number of classes
                        args.num_layers,                              # num_layers
                        args.emb_dim,                                 # hidden
                        dropout_rate=args.drop_rate,                  # dropout_rate
                        in_dropout_rate=args.indrop_rate,             # dropout rate for input
                        max_dim=dataset.max_dim,                      # max_dim
                        jump_mode=args.jump_mode,                     # jumping knowledge
                        nonlinearity=args.nonlinearity,               # nonlinearity
                        readout=args.readout,                         # readout
                        final_readout=args.final_readout,             # final readout
                        apply_dropout_before=args.drop_position,      # where to apply dropout
                        use_coboundaries=use_coboundaries,            # whether to use coboundaries
                        use_boundaries=use_boundaries,
                        include_down_adj=args.include_down_adj,
                        init_reduce=args.init_method,
                        graph_norm=args.graph_norm,                   # normalization layer
                        disable_graph_norm=disable_graph_norm,
                        readout_dims=args.readout_dims,               # readout_dims
                        num_layers_update=args.num_layers_update,     # number of update layers
                        num_layers_combine=args.num_layers_combine,   # number of combine layers
                        num_fc_layers=args.num_fc_layers,             # number of fc layers after msg prop
                        conv_type=args.conv_type,                     # conv type A or B
                        ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))
    return model

def get_graph_model(args, num_features, num_classes, device):
    # Instantite model
    if args.model == 'gin':
        model = GIN0(num_features,                            # num_input_features
                     args.num_layers,                         # num_layers
                     args.emb_dim,                            # hidden
                     num_classes,                             # num_classes
                     dropout_rate=args.drop_rate,             # dropout rate
                     nonlinearity=args.nonlinearity,          # nonlinearity
                     readout=args.readout,                    # readout
        ).to(device)
    else:
        raise ValueError('Invalid model type {}.'.format(args.model))
    compute_params(model)
    return model

def compute_params(model):
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("=================== Params stats ========================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")
    print("=========================================================")