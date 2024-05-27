import torch.optim as optim
from utils import *
# TODO: move following imports to utils?


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



