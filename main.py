from pathlib import Path
import json
import random
import datetime
import argparse
from collections import OrderedDict

from models.Hypernetworks import ShakesHyper, LoraHyper
from models.language_transformer import Transformer
from methods.method import *

from models.Bert import BertForSequenceClassification
from models.Lora import add_lora_layers, freeze_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='FedTP',
                        help='communication strategy: fedavg/FedTP/Personalized-T/pFedHN/pfedMe/fedprox/fedPer/fedBN/fedRod/fedproto/local_training/FedTP-Per/FedTP-Rod')
    parser.add_argument('--comm_round', type=int, default=1500, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='None', help='Noise type: None/increasng/space')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')
    parser.add_argument('--train_acc_pre', action='store_true')
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--test_round', type=int, default=1300)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--comment", default="_")
    parser.add_argument("--definite_selection", action='store_true')
    parser.add_argument("--show_all_accuracy", action='store_true')
    parser.add_argument("--version", type=int, default=1)

    """
    Used for FedTP
    """
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--hyper_hid', type=int, default=150, help="hypernet hidden dim")
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--balanced_soft_max", action='store_true')
    parser.add_argument("--client_embed_size", type=int, default=128)
    parser.add_argument('--log_flag', default=True)
    parser.add_argument('--k_neighbor', action='store_true')

    parser.add_argument('--capacity_ratio', type=float, default=1.0)
    parser.add_argument('--k_value', default=10)
    parser.add_argument('--knn_weight', type=float, default=0.6)

    """
    Used for shakespeare
    """
    parser.add_argument('--chunk_len', type=int, default=5)

    """
    Used for pfedMe
    """
    parser.add_argument('--pfedMe_k', type=int, default=5)
    parser.add_argument('--pfedMe_lambda', type=float, default=15)
    parser.add_argument('--pfedMe_beta', type=float, default=1)
    parser.add_argument('--pfedMe_mu', type=float, default=0)

    """
    Used for fedproto
    standard deviation: 2; rounds: 110; weight of proto loss: 0.1 
    local_bs 32

    """
    parser.add_argument('--fedproto_lambda', default=0.1)

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    device = torch.device(args.device)

    for net_i in range(n_parties):
        if args.model == "transformer":
            net = Transformer(n_src_vocab=len(string.printable),
                              n_trg_vocab=len(string.printable),
                              d_k=64, d_v=64, d_model=128,
                              d_word_vec=128, d_inner=256,
                              n_layers=args.depth, n_head=8, dropout=0.1)
        elif args.model == "bert-lora":
            bert_base = BertForSequenceClassification.from_pretrained(  # Pre-trained BERT
                model_type='bert-base-uncased',
                config_args={"vocab_size": 30522, "n_classes": 2}
                # these are default configs but just added for explicity
            )
            # TODO: add LoRA parameters to Args
            add_lora_layers(bert_base, ("query", "key", "value"), r=8, lora_alpha=16)  # add the LoRA layers into the model
            freeze_model(bert_base)  # freeze the non-LoRA parameters
            net = bert_base
        else:
            raise NotImplementedError("not supported yet")

        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_hyper(args, sam_node=None):
    # embed_dim = int(1 + args.n_parties / 4)
    embed_dim = args.client_embed_size

    if args.dataset == "shakespeare":
        hnet = ShakesHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid, dim=128,
                           heads=8, dim_head=64, n_hidden=args.n_hidden,
                           depth=args.depth, client_sample=sam_node)

    elif args.dataset == "stack_overflow_questions":
        hnet = LoraHyper(args.n_parties, embed_dim, hidden_dim=args.hyper_hid, bert_emb_dim=768,
                           lora_rank=8, n_hidden=args.n_hidden,
                           depth=args.depth, client_sample=sam_node)

    return hnet


if __name__ == '__main__':
    args = get_args()
    logging.info("Dataset: %s" % args.dataset)
    logging.info("Backbone: %s" % args.model)
    logging.info("Method: %s" % args.alg)
    logging.info("Partition: %s" % args.partition)
    logging.info("Beta: %f" % args.beta)
    logging.info("Sample rate: %f" % args.sample)
    logging.info("Print Accuracy on training set: %s" % args.train_acc_pre)
    logging.info("Save model: %s" % args.save_model)
    logging.info("Total running round: %s" % args.comm_round)
    logging.info("Test round fequency: %d" % args.eval_step)
    logging.info("Noise Type: %s" % args.noise_type)
    logging.info("Show every client's accuracy: %s" % args.show_all_accuracy)
    if args.noise_type != 'None':
        if args.partition != 'homo':
            raise NotImplementedError("Noise based feature skew only supports iid partition")
        logging.info("Max Noise: %d" % args.noise)
    if args.model in ["vit", "transformer"]:
        logging.info("Transformer depth: %d" % args.depth)
        if args.alg in ["FedTP", "FedTP-Rod"]:
            logging.info("Hyper hidden dimension: %d" % args.hyper_hid)
            logging.info("Client embedding size: %d" % args.client_embed_size)
            logging.info("Use balance soft-max: %s" % args.balanced_soft_max)
    if args.dataset == "shakespeare":
        if args.model not in ['lstm', 'transformer']:
            raise NotImplementedError("Serial data needs lstm, transformer as backbone.")
    if args.alg == "fedprox":
        logging.info("mu value: %f" % args.mu)
    if args.test_round <= 1:
        raise NotImplementedError("test round should be larger than 1")

    save_path = args.alg + "-" + args.model + "-" + str(
        args.n_parties) + "-" + args.dataset + "-" + args.partition + args.comment
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    if args.k_neighbor:
        logging.info("Use memory: %s" % args.k_neighbor)

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path = args.alg + " " + args.model + " " + str(
            args.version) + '-experiment_arguments-%s.json ' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = args.model + " " + str(args.version) + '-experiment_log-%s ' % (
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    eval_step = args.eval_step
    acc_all = []

    logger.info("Partitioning data")
    if args.dataset == 'shakespeare':
        if args.model not in ["lstm", "transformer"]:
            raise NotImplementedError("shakespeare supports lstm or transformer")
        data_dir = os.path.join("data", "shakespeare", "all_data", "train")
        train_dl_global, val_dl_global, test_dl_global, original_c_num = get_spe_dataloaders(args.dataset, data_dir,
                                                                                             args.batch_size,
                                                                                             args.chunk_len)
        args.n_parties = len(train_dl_global)
    elif args.dataset == 'stack_overflow_questions':
        #data_dir = './data/stack_overflow_questions/'
        data_dir = os.path.join("data", "stack_overflow_dataset", "train")
        train_dl_global, val_dl_global, test_dl_global, original_c_num = get_spe_dataloaders(args.dataset, data_dir,
                                                                                             args.batch_size,
                                                                                             args.chunk_len)

    logging.info("Test beginning round: %d" % args.test_round)
    logger.info("Drop Client Number: %d" % (original_c_num - args.n_parties))
    logger.info("Client Number: %d" % args.n_parties)
    logger.info("Chunk_len: %d" % args.chunk_len)

    results_dict = defaultdict(list)
    eval_step = args.eval_step
    best_step = 0
    best_accuracy = -1
    test_round = args.test_round

    if args.alg == 'fedavg':
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()  # TODO: verify that only LoRA layers are set here

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # train local models
            if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in
                                 range(len(instance_number_per_client))]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()  # TODO: verify here as well that only LoRA layers are taken
                if idx == 0:
                    for key in net_para:
                        if args.model == "bert-lora" and "lora" not in key and "classifier" not in key:
                            continue
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        if args.model == "bert-lora" and "lora" not in key and "classifier" not in key:
                            continue
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            if (round + 1) >= test_round and (round + 1) % eval_step == 0:
                if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client_simple(
                        global_model, args, train_dl_global, test_dl_global, nets, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' % test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc * 100)

        save_path = Path("results_table/" + save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(
            args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_gmodel = os.path.join(save_path, 'gmodel_1500.tar')
            torch.save({'epoch': args.comm_round + 1, 'state': global_model.state_dict()}, outfile_gmodel)

        json_file_opt = "results_" + accessories + ".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)

    elif args.alg == 'FedTP':
        if args.model not in ["transformer", "bert-lora"]:
            raise NotImplementedError("FedTP only supports ViT and transformer")

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        logger.info("Initializing hyper")
        if args.dataset == "shakespeare" or args.dataset == "stack_overflow_questions":
            sam_node = int(args.n_parties * args.sample)
            hnet = init_hyper(args, sam_node).to(device)

        optimizers = {
            'sgd': torch.optim.SGD(
                params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
            ),
            'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr)
        }
        optimizer = optimizers['sgd']

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round: %d" % round)

            hnet.train()
            grads_update = []

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            weights = hnet(torch.tensor([selected], dtype=torch.long).to(device), False)

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for ix in range(len(selected)):
                        node_weights = weights[ix]
                        idx = selected[ix]
                        nets[idx].load_state_dict(global_para)
                        nets[idx].load_state_dict(node_weights, strict=False)
                        del node_weights
            else:
                for ix in range(len(selected)):
                    node_weights = weights[ix]
                    idx = selected[ix]
                    nets[idx].load_state_dict(global_para)
                    nets[idx].load_state_dict(node_weights, strict=False)
                    del node_weights

            if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                nets_list = local_train_net_per(nets, selected, args, train_dl_global, test_dl_global, logger, device=device)

            # update global model
            if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                instance_number_per_client = [len(train_dl_global[r].dataset) for r in selected]
                total_data_points = sum(instance_number_per_client)
                fed_avg_freqs = [instance_number_per_client[r] / total_data_points for r in
                                 range(len(instance_number_per_client))]

            for idx in range(len(selected)):
                final_state = nets[selected[idx]].state_dict()
                net_para = nets[selected[idx]].state_dict()

                node_weights = weights[idx]
                inner_state = OrderedDict({k: tensor.data for k, tensor in node_weights.items()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in node_weights.keys()})
                hnet_grads = torch.autograd.grad(
                    list(node_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()),
                    retain_graph=True
                )

                if idx == 0:
                    for key in net_para:
                        if args.model == "bert-lora" and "lora" not in key and "classifier" not in key:
                            continue
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    grads_update = [fed_avg_freqs[idx] * x for x in hnet_grads]
                else:
                    for key in net_para:
                        if args.model == "bert-lora" and "lora" not in key and "classifier" not in key:
                            continue
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
                    for g in range(len(hnet_grads)):
                        grads_update[g] += fed_avg_freqs[idx] * hnet_grads[g]
                del final_state, net_para, node_weights, inner_state, delta_theta, hnet_grads

            global_model.load_state_dict(global_para)
            optimizer.zero_grad()
            for p, g in zip(hnet.parameters(), grads_update):
                p.grad = g
            optimizer.step()

            del grads_update, weights, global_para, nets_list

            if (round + 1) >= test_round and (round + 1) % eval_step == 0:
                if args.dataset == 'shakespeare' or args.dataset == 'stack_overflow_questions':
                    train_results, train_avg_loss, train_acc, train_all_acc, test_results, test_avg_loss, test_acc, test_all_acc = compute_accuracy_per_client(
                        hnet, nets, global_model, args, train_dl_global, test_dl_global, 0, device=device)

                if args.log_flag:
                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)
                    logger.info('>> Test avg loss: %f' % test_avg_loss)

                results_dict['train_avg_loss'].append(train_avg_loss)
                results_dict['train_avg_acc'].append(train_acc)
                results_dict['test_all_acc'] = test_all_acc
                results_dict['test_avg_loss'].append(test_avg_loss)
                results_dict['test_avg_acc'].append(test_acc * 100)

        save_path = Path("results_table/" + save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        accessories = args.alg + "-" + str(args.n_parties) + "-" + str(
            args.dataset) + "-" + args.partition + "-" + args.comment

        if args.save_model:
            logger.info("Saving model")
            outfile_hp = os.path.join(save_path, 'HY_1500.tar')
            # outfile_vit = os.path.join(save_path, 'Vit_1500.tar')
            torch.save({'epoch': args.comm_round + 1, 'state': hnet.state_dict()}, outfile_hp)
            # torch.save({'epoch': args.comm_round + 1, 'state': global_model.state_dict()}, outfile_vit)

        json_file_opt = "results_" + accessories + ".json"
        with open(str(save_path / json_file_opt), "w") as file:
            json.dump(results_dict, file, indent=4)


    acc_all = np.asarray(results_dict['test_avg_acc'])
    logger.info("Accuracy Record: ")
    logger.info(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    logger.info('Test Acc = %4.2f%% +- %4.2f%%' % (acc_mean, acc_std))
    if args.show_all_accuracy:
        logger.info("Accuracy in each client: ")
        logger.info(results_dict['test_all_acc'])
