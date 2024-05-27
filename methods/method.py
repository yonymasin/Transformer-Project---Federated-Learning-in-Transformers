import torch.optim as optim
from utils import *
# TODO: move following imports to utils?
from pathlib import Path
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer

def train_net_shakes(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None, fedprox=False):

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    optimizer = optim.SGD(
            [param for param in net.parameters() if param.requires_grad],
            lr=lr, momentum=0., weight_decay=5e-4)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    net.train()
    global_loss = 0.
    global_metric = 0.
    n_samples = 0
    for epoch in range(epochs):
        for tmp in train_dataloader:
            for x, y, indices in tmp:
                x = x.to(device)
                y = y.to(device)

                n_samples += y.size(0)
                chunk_len = y.size(1)
                optimizer.zero_grad()

                y_pred, _ = net(x)
                loss_vec = criterion(y_pred, y)
                loss = loss_vec.mean()

                loss.backward()
                optimizer.step()
                global_loss += loss.item() * loss_vec.size(0) / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

                del y_pred, loss, loss_vec, x, y, predicted, acc, correct

    te_metric, te_samples, _ = compute_accuracy_shakes(net, test_dataloader, device=device)
    test_acc = te_metric/te_samples

    if args.train_acc_pre:
        tr_metric, tr_samples, _ = compute_accuracy_shakes(net, train_dataloader, device=device)
        train_acc = tr_metric/tr_samples
        return train_acc, test_acc
    else:
        return None, test_acc


def train_net_bert(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None):
    tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
    trainer_bert_base_lora = BertTrainer(
        net,
        net_id,
        tokenizer_base,
        lr=lr,
        epochs=epochs,
        optimizer_type=args_optimizer,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        output_dir=f"./trained_models/net_{net_id}_bert_fine_tuned_lora",
        output_filename='bert_base_lora',
        save=True,
    )
    # TODO: do we want to save accuracies, F1, Recall, Precision and Loss statistics during training. If so need to add support
    train_acc, test_acc = trainer_bert_base_lora.train(evaluate=True)

    if args.train_acc_pre:
        return train_acc, test_acc
    else:
        return None, test_acc

def local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc = train_net_shakes(net_id, net, train_dl_local, test_dl_local,
                n_epoch, args.lr, args.optimizer, args, device=device)
        if args.dataset == "stack_overflow_questions":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc = train_net_bert(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        del trainacc, testacc, train_dl_local, test_dl_local, net

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    nets_list = list(nets.values())

    del avg_acc, nets

    return nets_list


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
