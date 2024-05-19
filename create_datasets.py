import torch.utils.data as data
import torch
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from typing import Optional, Callable
from torch.utils.model_zoo import tqdm
import pandas as pd

import os.path
import logging
import json

import torch.utils.data as torch_data
import datasets
from constants import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:  # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")


class CharacterDataset(torch_data.Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input

        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences

        """
        self.all_characters = string.printable

        # self.all_characters:
        # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx + self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx + 1:idx + self.chunk_len + 1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


# define a function to apply standard scaling to the tabular data
def standard_scale(example, mean_train, std_train):
    example['tabular'] = (torch.tensor(example['tabular']) - mean_train) / std_train
    return example


def create_stack_overflow_questions_dataset(path):

    df = pd.read_csv(path)
    df['OpenStatusInt'] = df['OpenStatus'].map(SOQ_LABEL_STRING_TO_INT)  # convert class strings to integers
    df['BodyLength'] = df['BodyMarkdown'].apply(lambda x: len(x.split(" ")))  # number of words in body text
    df['TitleLength'] = df['Title'].apply(lambda x: len(x.split(" ")))  # number of words in title text
    df['TitleConcatWithBody'] = df.apply(lambda x: x.Title + " " + x.BodyMarkdown,
                                             axis=1)  # combine title and body text
    df['NumberOfTags'] = df.apply(
        lambda x: len([x[col] for col in ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'] if not pd.isna(x[col])]),
        axis=1,
    )  # number of tags
    df['PostCreationDate'] = pd.to_datetime(df['PostCreationDate'])  # convert string to Timedelta object
    df['OwnerCreationDate'] = pd.to_datetime(df['OwnerCreationDate'],
                                                 format='mixed')  # convert string to Timedelta object
    df['DayDifference'] = (df['PostCreationDate'] - df[
        'OwnerCreationDate']).dt.days  # days between account creation and post creation
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
    dataset_stackoverflow = datasets.Dataset.from_dict(data_dict)

    # calculate mean and std of each tabular feature
    mean_train = torch.mean(torch.tensor(dataset_stackoverflow['tabular'], dtype=torch.float32), dim=0)
    std_train = torch.std(torch.tensor(dataset_stackoverflow['tabular'], dtype=torch.float32), dim=0)

    # apply the standard scaling function to the tabular features
    dataset_stackoverflow = dataset_stackoverflow.map(lambda example: standard_scale(example, mean_train, std_train))

    ##################################################################################
    # # TODO: figure out how to create categorical groups for the Stack Overflow tags (use openai API? with sampling? number of groups?)
    # for tag_idx in tqdm.tqdm(range(1, NUM_TAGS + 1), desc='Creating tags dictionaries'):
    #     if not os.path.exists(f"./Tag{tag_idx}Hist.json"):
    #         unique_vals = df[f"Tag{tag_idx}"].unique()
    #         df_hist = {}
    #         for unique_val in unique_vals:
    #             df_hist[unique_val] = (sum([1 if val == unique_val else 0 for val in df[f"Tag{tag_idx}"]]))
    #         # Convert and write JSON object to file
    #         with open(f"Tag{tag_idx}Hist.json", "w") as outfile:
    #             json.dump(df_hist, outfile)
    #
    # if not os.path.exists(f"./AllTagsHist.json"):
    #     full_dict = {}
    #     for tag_idx in tqdm.tqdm(range(1, NUM_TAGS + 1), desc='Merging tags dictionaries'):
    #         with open(f"Tag{tag_idx}Hist.json", "r") as file:
    #             df_hist = json.load(file)
    #         if tag_idx == 1:
    #             full_dict = df_hist
    #             continue
    #         else:
    #             full_dict = {k: full_dict.get(k, 0) + df_hist.get(k, 0) for k in set(full_dict) | set(df_hist)}
    #
    #     with open(f"AllTagsHist.json", "w") as outfile:
    #         json.dump(full_dict, outfile)
    #
    # with open(f"AllTagsHist.json", "r") as file:
    #     full_dict = json.load(file)
    #
    # print(full_dict.values())
    # print(full_dict.keys())
    # print(max(full_dict.values()))
    # v = list(full_dict.values())
    # k = list(full_dict.keys())
    # print(k[v.index(min(v))])
    ##################################################

    return dataset_stackoverflow