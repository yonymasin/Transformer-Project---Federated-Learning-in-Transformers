# Transformer-Project---Federated-Learning-in-Transformers
A project for Transformers course, researching new ideas in training Transformers in a federated learning setting
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Data Pre-Processing
Run `python preprocess_shakespeare.py all_data/all_data.txt all_data/` from `data/shakespeare/`.
## Usage
Here is one example to run FedTP:
```
python main.py --model=transformer --dataset=shakespeare --alg=FedTP  --lr=0.01 --batch-size=64 --epochs=5 --n_parties=10 --rho=0.9 --comm_round=300 --device=cuda:0 --datadir='./data/' --logdir='./logs_emb/'  --init_seed=0 --chunk_len=1 --sample=0.1 --test_round=250 
```

Full parameter description is presented in following table:

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `cnn`, `cnn-b`, `vit`, `lstm`, `transformer`. Default = `vit`. |
| `dataset`      | Dataset to use. Options: `cifar10`, `cifar100`, `shakespeare`. Default = `cifar10`. |
| `alg` | Basic training algorithm. Basic Options: `fedavg`, `fedprox`, `FedTP`, `pFedHN`, `pfedMe`, `fedPer`, `fedBN`, `fedRod`, `fedproto`, `local_training`. Extension: `Personalized-T`, `FedTP-Per`, `FedTP-Rod`. Default = `FedTP`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `1`. |
| `n_parties` | Number of parties, default = `10`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `eval_step`    | Test interval during communication, default = `1`. |
| `test_round`    | Round beginning to test, default = `2`. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-labeldir100`, `noniid-labeluni`, `iid-label100`, `homo`. Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `sample` | Ratio of parties that participate in each communication round, default = `0.1`. |
| `balanced_soft_max` | Activate this to run FedRod and FedTP-Rod. |
| `k_neighbor` | Activate this to run FedTP-KNN. |
| `init_seed` | The initial seed, default = `0`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `noise_type` | Noise type. Use `increasing` to check effect of heterogeneity in Noise-based Feature Imbalance, default = `None`. |
| `save_model` | Activate this to save model. |
