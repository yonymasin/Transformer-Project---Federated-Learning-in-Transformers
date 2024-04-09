# import sys
# sys.path.insert(0, '../src')

# standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import math

# related third party imports
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from bert_from_scratch import MyBertForSequenceClassification



##############################################################
##                   Data Pre-Processing                    ##
##############################################################
def collate_fn(batch):
	""" Instructs how the DataLoader should process the data into a batch"""
    
    text = [item['text'] for item in batch]
    tabular = torch.stack([torch.tensor(item['tabular']) for item in batch])
    labels = torch.stack([torch.tensor(item['label']) for item in batch])

    return {'text': text, 'tabular': tabular, 'label': labels}


# define a function to apply standard scaling to the tabular data
def standard_scale(example):
    example['tabular'] = (torch.tensor(example['tabular']) - mean_train) / std_train
    return example

def preprocess_sof_data(data_csv_path="./data/stack_overflow_questions/train-sample.csv"):
	df = pd.read_csv(data_csv_path)

	# dict mapping strings to integers
	string_to_int = {
    	'open': 0,
    	'not a real question': 1,
   		'off topic': 1,
   		'not constructive': 1,
   		'too localized': 1		
   	}

    # add new features to dataframe
	df['OpenStatusInt'] = df['OpenStatus'].map(string_to_int)  # convert class strings to integers		
	df['BodyLength'] = df['BodyMarkdown'].apply(lambda x: len(x.split(" ")))  # number of words in body text
	df['TitleLength'] = df['Title'].apply(lambda x: len(x.split(" ")))  # number of words in title text
	df['TitleConcatWithBody'] = df.apply(lambda x: x.Title +  " " + x.BodyMarkdown, axis=1)  # combine title and body text
	df['NumberOfTags'] = df.apply(
   		lambda x: len([x[col] for col in ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'] if not pd.isna(x[col])]), 
    	axis=1,
	)  # number of tags
	df['PostCreationDate'] = pd.to_datetime(df['PostCreationDate'])  # convert string to Timedelta object
	df['OwnerCreationDate'] = pd.to_datetime(df['OwnerCreationDate'], format='mixed')  # convert string to Timedelta object
	df['DayDifference'] = (df['PostCreationDate'] - df['OwnerCreationDate']).dt.days  # days between account creation and post creation 

	# list of col names with tabular data 
	tabular_feature_list = [
    	'ReputationAtPostCreation',  
    	'BodyLength', 
    	'TitleLength', 
    	'NumberOfTags',
    	'DayDifference',
	]

	# place the desired data from the dataframe into a dictionary
	data_dict = {
	    'text': df.TitleConcatWithBody.tolist(),
	    'tabular': df[tabular_feature_list].values,
	    'label': df.OpenStatusInt.tolist(),
	}
	
	# load data into hugging face dataset object
	dataset_stackoverflow = Dataset.from_dict(data_dict)
	
	# define the indices at which to split the dataset into train/validation/test
	n_samples = len(dataset_stackoverflow)
	split_idx1 = int(n_samples * 0.8)
	split_idx2 = int(n_samples * 0.9)
	
	# shuffle the dataset
	shuffled_dataset = dataset_stackoverflow.shuffle(seed=42)
	
	# split dataset training/validation/test
	train_dataset = shuffled_dataset.select(range(split_idx1))
	val_dataset = shuffled_dataset.select(range(split_idx1, split_idx2))
	test_dataset = shuffled_dataset.select(range(split_idx2, n_samples))
	
	# calculate mean and std of each tabular feature
	mean_train = torch.mean(torch.tensor(train_dataset['tabular'], dtype=torch.float32), dim=0)
	std_train = torch.std(torch.tensor(train_dataset['tabular'], dtype=torch.float32), dim=0)
	
	
	
	# apply the standard scaling function to the tabular features
	train_dataset = train_dataset.map(standard_scale)
	val_dataset = val_dataset.map(standard_scale)
	test_dataset = test_dataset.map(standard_scale)
	
	# load the datasets into a dataloader
	train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
	val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
	test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

	return train_dataloader, val_dataloader, test_dataloader


##############################################################
##                   BERT Trainer Class                     ##
##############################################################
class BertTrainer:
    """ A training and evaluation loop for PyTorch models with a BERT like architecture. """
    
    
    def __init__(
        self, 
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        epochs=1,
        lr=5e-04,
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
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.save = save
        self.eval_loss = float('inf')  # tracks the lowest loss so as to only save the best model  
        self.epochs = epochs
        self.epoch_best_model = 0  # tracks which epoch the lowest loss is in so as to only save the best model
        self.tabular = tabular
    
    def train(self, evaluate=False):
        """ Calls the batch iterator to train and optionally evaluate the model."""
        for epoch in range(self.epochs):
            self.iteration(epoch, self.train_dataloader)
            if evaluate and self.eval_dataloader is not None:
                self.iteration(epoch, self.eval_dataloader, train=False)

    def evaluate(self):
        """ Calls the batch iterator to evaluate the model."""
        epoch=0
        self.iteration(epoch, self.eval_dataloader, train=False)
    
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
            f"samples={samples_accumulated}, \
            correct={correct_accumulated}, \
            acc={round(accuracy, 4)}, \
            recall={round(recall, 4)}, \
            prec={round(precision,4)}, \
            f1={round(f1, 4)}, \
            loss={round(avg_loss_epoch, 4)}"
        )    
        
        # save the model if the evaluation loss is lower than the previous best epoch 
        if self.save and not train and avg_loss_epoch < self.eval_loss:
            
            # create directory and filepaths
            dir_path = Path(self.output_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{self.output_filename}_epoch_{epoch}.pt"
            
            # delete previous best model from hard drive
            if epoch > 0:
                file_path_best_model = dir_path / f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
                !rm -f $file_path_best_model
            
            # save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)
            
            # update the new best loss and epoch
            self.eval_loss = avg_loss_epoch
            self.epoch_best_model = epoch



##############################################################
##                      BERT Inference                      ##
##############################################################

def test_bert(train_dataloader, test_dataloader):
	
	# load tokenizer and pretrained model
	tokenizer_base = BertTokenizer.from_pretrained('bert-base-uncased')
	bert_base = MyBertForSequenceClassification.from_pretrained(
	    model_type='bert-base-uncased',
	    config_args={"vocab_size": 30522, "n_classes": 2}  # these are default configs but just added for explicity
	)
	
	# Get trainer
	# Note: We only evaluate so lr and epochs are 'DONT CARE'
	trainer_bert_base = BertTrainer(
	    bert_base,
	    tokenizer_base,
	    lr=5e-06,
	    epochs=5,
	    train_dataloader=train_dataloader,
	    eval_dataloader=test_dataloader,
	    output_dir='../models/bert_base_fine_tuned',
	    output_filename='bert_base',
	    save=False,
	)
	
	# evaluate on test set
	trainer_bert_base.evaluate()

if __name__ == '__main__':
	train_dataloader, val_dataloader, test_dataloader = preprocess_sof_data(path_to_data)
	test_bert(train_dataloader, test_dataloader)