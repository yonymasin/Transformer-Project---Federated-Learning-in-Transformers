import logging
import numpy as np
import torch
import torch.utils.data as data
import copy
from collections import defaultdict
import torch.nn as nn
from transformers import BertTokenizer
from pathlib import Path
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

class BertTrainer:
    """ A training and evaluation loop for PyTorch models with a BERT like architecture. """

    def __init__(
            self,
            model,
            model_id,
            tokenizer,
            train_dataloader,
            eval_dataloader=None,
            epochs=1,
            lr=5e-04,
            optimizer_type='AdamW',
            device='cpu',
            output_dir='./',
            output_filename='model_state_dict.pt',
            save=False,
            tabular=False,
    ):
        """
        Args:
            model: torch.nn.Module: = A PyTorch model with a BERT like architecture,
            tokenizer: = A BERT tokenizer for tokenizing text input,
            train_dataloader: torch.utils.data.DataLoader =
                A dataloader containing the training data with "text" and "label" keys (optionally a "tabular" key),
            eval_dataloader: torch.utils.data.DataLoader =
                A dataloader containing the evaluation data with "text" and "label" keys (optionally a "tabular" key),
            epochs: int = An integer representing the number epochs to train,
            lr: float = A float representing the learning rate for the optimizer,
            output_dir: str = A string representing the directory path to save the model,
            output_filename: string = A string representing the name of the file to save in the output directory,
            save: bool = A boolean representing whether or not to save the model,
            tabular: bool = A boolean representing whether or not the BERT model is modified to accept tabular data,
        """

        self.device = device
        self.model = model.to(self.device)
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        if optimizer_type == 'adamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        else:
            raise NotImplementedError("BERT fine-tuning only supports AdamW, Adam and SGD optimizers")
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.save = save
        self.eval_loss = float('inf')  # tracks the lowest loss so as to only save the best model
        self.eval_acc = 0.0  # tracks the highest accuracy so as to only save the best model
        self.train_acc = []
        self.epochs = epochs
        self.epoch_best_model = 0  # tracks which epoch the lowest loss is in so as to only save the best model
        self.tabular = tabular

    def calc_avg_train_acc(self):
        acc_sum = 0
        for acc in self.train_acc:
            acc_sum += acc
        return acc_sum / len(self.train_acc)

    def train(self, evaluate=False):
        """ Calls the batch iterator to train and optionally evaluate the model."""
        for epoch in range(self.epochs):
            self.iteration(epoch, self.train_dataloader)
            if evaluate and self.eval_dataloader is not None:
                self.iteration(epoch, self.eval_dataloader, train=False)

        return self.calc_avg_train_acc(), self.eval_acc

    def evaluate(self):
        """ Calls the batch iterator to evaluate the model."""
        epoch = 0
        self.iteration(epoch, self.eval_dataloader, train=False)
        return self.eval_acc, self.eval_loss

    def iteration(self, epoch, data_loader, train=True):
        """ Iterates through one epoch of training or evaluation"""

        # initialize variables
        loss_accumulated = 0.
        correct_accumulated = 0
        samples_accumulated = 0
        preds_all = []
        labels_all = []

        self.model.train() if train else self.model.eval()

        # progress bar
        mode = "train" if train else "eval"
        batch_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EP ({mode}) {epoch}",
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        # iterate through batches of the dataset
        for i, batch in batch_iter:

            # tokenize data
            batch_t = self.tokenizer(
                batch['text'],
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt',
            )
            batch_t = {key: value.to(self.device) for key, value in batch_t.items()}
            batch_t["input_labels"] = batch["label"].to(self.device)
            batch_t["tabular_vectors"] = batch["tabular"].to(self.device)

            # forward pass - include tabular data if it is a tabular model
            if self.tabular:
                logits = self.model(
                    input_ids=batch_t["input_ids"],
                    token_type_ids=batch_t["token_type_ids"],
                    attention_mask=batch_t["attention_mask"],
                    tabular_vectors=batch_t["tabular_vectors"],
                )

            else:
                logits = self.model(
                    input_ids=batch_t["input_ids"],
                    token_type_ids=batch_t["token_type_ids"],
                    attention_mask=batch_t["attention_mask"],
                )

            # calculate loss
            loss = self.loss_fn(logits, batch_t["input_labels"])

            # compute gradient and and update weights
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # calculate the number of correct predictions
            preds = logits.argmax(dim=-1)
            correct = preds.eq(batch_t["input_labels"]).sum().item()

            # accumulate batch metrics and outputs
            loss_accumulated += loss.item()
            correct_accumulated += correct
            samples_accumulated += len(batch_t["input_labels"])
            preds_all.append(preds.detach())
            labels_all.append(batch_t['input_labels'].detach())

        # concatenate all batch tensors into one tensor and move to cpu for compatibility with sklearn metrics
        preds_all = torch.cat(preds_all, dim=0).cpu()
        labels_all = torch.cat(labels_all, dim=0).cpu()

        # metrics
        accuracy = accuracy_score(labels_all, preds_all)
        precision = precision_score(labels_all, preds_all, average='macro')
        recall = recall_score(labels_all, preds_all, average='macro')
        f1 = f1_score(labels_all, preds_all, average='macro')
        avg_loss_epoch = loss_accumulated / len(data_loader)

        # print metrics to console
        print(
            f"Net {self.model_id}, \
            Samples accumulated = {samples_accumulated}, \
            Correct = {correct_accumulated}, \
            Accuracy = {round(accuracy, 4)}, \
            Recall = {round(recall, 4)}, \
            Precision = {round(precision, 4)}, \
            F1 = {round(f1, 4)}, \
            Loss = {round(avg_loss_epoch, 4)}"
        )

        if train:
            self.train_acc.append(round(accuracy, 4))

        # save the model if the evaluation loss is lower than the previous best epoch
        if self.save and not train and avg_loss_epoch < self.eval_loss:

            # create directory and filepaths
            dir_path = Path(self.output_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{self.output_filename}_epoch_{epoch}.pt"

            # delete previous best model from hard drive
            if epoch > 0:
                file_path_best_model = dir_path / f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
                os.remove(file_path_best_model)

            # save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)

            # update the new best loss and epoch
            self.eval_loss = avg_loss_epoch
            self.eval_acc = round(accuracy, 4)
            self.epoch_best_model = epoch

