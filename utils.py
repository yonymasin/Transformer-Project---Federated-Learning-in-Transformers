import logging
import numpy as np
import torch
import torch.utils.data as data
import copy
from collections import defaultdict
import torch.nn as nn
from transformers import BertTokenizer

from create_datasets import CharacterDataset, create_stack_overflow_questions_dataset
from constants import *
from methods import BertTrainer


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def record_net_data_stats(y_train, net_dataidx_map, logdir=None):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    if logdir != None:
        logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def compute_accuracy_shakes(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    # true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    # correct, total = 0, 0
    global_loss = 0.
    global_metric = 0.
    n_samples = 0

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    with torch.no_grad():
        for tmp in dataloader:
            for x, y, indices in tmp:
                logger.info(x)
                x = x.to(device)
                y = y.to(device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred, _ = model(x)
                global_loss += criterion(y_pred, y).sum().item() / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

    if was_training:
        model.train()

    return global_metric, n_samples, global_loss / n_samples


def compute_accuracy_stack_overflow_questions(model, model_id, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    n_samples = len(dataloader.dataset)

    tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
    trainer_bert_base_lora = BertTrainer(
        model,
        model_id,
        tokenizer_base,
        lr=None,
        epochs=0,
        optimizer_type=None,
        device=device,
        train_dataloader=dataloader,
        eval_dataloader=dataloader,
        output_dir=f"./eval_models/net_{model_id}_bert_fine_tuned_lora",
        output_filename='bert_base_lora',
        save=True,
    )
    
    eval_acc, eval_loss = trainer_bert_base_lora.evaluate()

    if was_training:
        model.train()

    return eval_acc * n_samples, n_samples, eval_loss / n_samples


def compute_accuracy_per_client_simple(global_model, args, net_dataidx_map_train, net_dataidx_map_test, nets=None,
                                       device="cpu"):
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))
    for net_id in range(args.n_parties):

        local_model = copy.deepcopy(global_model)
        local_model.eval()

        if args.dataset == 'shakespeare':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local,
                                                                                     device=device)

        if args.dataset == 'stack_overflow_questions':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_stack_overflow_questions(local_model, net_id, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_stack_overflow_questions(local_model, net_id, train_dl_local,
                                                                                     device=device)

        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]

    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def compute_accuracy_per_client(hyper, nets, global_model, args, net_dataidx_map_train, net_dataidx_map_test, num_class,
                                device="cpu"):
    hyper.eval()
    if args.train_acc_pre:
        train_results = defaultdict(lambda: defaultdict(list))
    test_results = defaultdict(lambda: defaultdict(list))

    for net_id in range(args.n_parties):

        node_weights = hyper(torch.tensor([net_id], dtype=torch.long).to(device), True)
        local_model = copy.deepcopy(global_model)
        local_model.load_state_dict(node_weights, strict=False)
        local_model.eval()

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_shakes(local_model, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_shakes(local_model, train_dl_local,
                                                                                     device=device)

        if args.dataset == 'stack_overflow_questions':
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]

            test_correct, test_total, test_avg_loss = compute_accuracy_stack_overflow_questions(local_model, net_id, test_dl_local, device=device)
            if args.train_acc_pre:
                train_correct, train_total, train_avg_loss = compute_accuracy_stack_overflow_questions(local_model, net_id, train_dl_local,
                                                                                     device=device)
        if args.train_acc_pre:
            train_results[net_id]['loss'] = train_avg_loss
            train_results[net_id]['correct'] = train_correct
            train_results[net_id]['total'] = train_total

        test_results[net_id]['loss'] = test_avg_loss
        test_results[net_id]['correct'] = test_correct
        test_results[net_id]['total'] = test_total

    test_total_correct = sum([val['correct'] for val in test_results.values()])
    test_total_samples = sum([val['total'] for val in test_results.values()])
    test_avg_loss = np.mean([val['loss'] for val in test_results.values()])
    test_avg_acc = test_total_correct / test_total_samples

    test_all_acc = [val['correct'] / val['total'] for val in test_results.values()]
    if args.train_acc_pre:
        train_total_correct = sum([val['correct'] for val in train_results.values()])
        train_total_samples = sum([val['total'] for val in train_results.values()])
        train_avg_loss = np.mean([val['loss'] for val in train_results.values()])
        train_acc_pre = train_total_correct / train_total_samples

        train_all_acc = [val['correct'] / val['total'] for val in train_results.values()]
        return train_results, train_avg_loss, train_acc_pre, train_all_acc, test_results, test_avg_loss, test_avg_acc, test_all_acc
    else:
        return 0, 0, 0, 0, test_results, test_avg_loss, test_avg_acc, test_all_acc


def get_spe_dataloaders(dataset, data_dir, batch_size, chunk_len, is_validation=False):
    inputs, targets = None, None
    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(os.listdir(data_dir)):
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = get_spe_loader(dataset_name=dataset,
                                        path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
                                        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets,
                                        train=True)

        val_iterator = get_spe_loader(dataset_name=dataset,
                                      path=os.path.join(task_data_path, f"train{EXTENSIONS[dataset]}"),
                                      batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets,
                                      train=False)

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator = get_spe_loader(dataset_name=dataset,
                                       path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[dataset]}"),
                                       batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets,
                                       train=False)

        if test_iterator != None:
            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    original_client_num = task_id + 1 # TODO: for Stack Overflow Questions dataset need to think how to compute num of clients

    return train_iterators, val_iterators, test_iterators, original_client_num


def collate_fn(batch):
    """ Instructs how the DataLoader should process the data into a batch"""

    text = [item['text'] for item in batch]
    tabular = torch.stack([torch.tensor(item['tabular']) for item in batch])
    labels = torch.stack([torch.tensor(item['label']) for item in batch])

    return {'text': text, 'tabular': tabular, 'label': labels}


def get_spe_loader(dataset_name, path, batch_size, train, chunk_len=5, inputs=None, targets=None):
    if dataset_name == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=chunk_len)
    elif dataset_name == 'stack_overflow_questions':
        dataset = create_stack_overflow_questions_dataset(path)
    else:
        raise NotImplementedError(f"{dataset_name} not recognized type")

    if len(dataset) == 0:
        return

    drop_last = (len(dataset) > batch_size) and train
    current_collate_fn = collate_fn if dataset_name == 'stack_overflow_questions' else None
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS, collate_fn=current_collate_fn)

