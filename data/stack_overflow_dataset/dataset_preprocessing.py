import pandas as pd
from sklearn.utils import shuffle
import os

def create_stack_overflow_questions_dataset(dataset_path):
    SOQ_LABEL_STRING_TO_INT = {
        'open': 0,
        'not a real question': 1,
        'off topic': 1,
        'not constructive': 1,
        'too localized': 1
    }
    df = pd.read_csv(dataset_path)
    df['OpenStatusInt'] = df['OpenStatus'].map(SOQ_LABEL_STRING_TO_INT)  # convert class strings to integers
    df['BodyLength'] = df['BodyMarkdown'].apply(lambda x: len(x.split(" ")))  # number of words in body text
    df['TitleLength'] = df['Title'].apply(lambda x: len(x.split(" ")))  # number of words in title text
    df['TitleConcatWithBody'] = df.apply(lambda x: x.Title + " " + x.BodyMarkdown,
                                             axis=1)  # combine title and body text
    df['NumberOfTags'] = df.apply(
        lambda x: len([x[col] for col in ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'] if not pd.isna(x[col])]),
        axis=1,
    )  # number of tags
    
    # list of col names with tabular data
    tabular_feature_list = [
        'ReputationAtPostCreation',
        'BodyLength',
        'TitleLength',
        'NumberOfTags',
    ]

    ##################################################################################
    # TODO: figure out how to create categorical groups for the Stack Overflow tags (use openai API? with sampling? number of groups?)
    # for tag_idx in tqdm.tqdm(range(1, NUM_TAGS + 1), desc='Creating tags dictionaries'):
    #     if not os.path.exists(f"./Tag{tag_idx}Hist.json"):
    #         unique_vals = df[f"Tag{tag_idx}"].unique()
    #         df_hist = {}
    #         for unique_val in unique_vals:
    #             df_hist[unique_val] = (sum([1 if val == unique_val else 0 for val in df[f"Tag{tag_idx}"]]))
    #         # Convert and write JSON object to file
    #         with open(f"Tag{tag_idx}Hist.json", "w") as outfile:
    #             json.dump(df_hist, outfile)
    
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
    
    #     with open(f"AllTagsHist.json", "w") as outfile:
    #         json.dump(full_dict, outfile)
    
    # with open(f"Tag1Hist.json", "r") as file:
    #     tag1_dict = json.load(file)
    
    unique_vals = df["Tag1"].unique()
    tag1_dict = {}
    for unique_val in unique_vals:
        tag1_dict[unique_val] = (sum([1 if val == unique_val else 0 for val in df["Tag1"]]))

    N_PARTIES = 100
    tag1_all_values = list(tag1_dict.values())
    tag1_all_values.sort(reverse=True)
    value_threshold = tag1_all_values[N_PARTIES-1]
    trimmed_tag1_dict = {key:val for key,val in tag1_dict.items() if val >= value_threshold}
    client_df_list = []
    client_num = 0
    for (client_num,key) in enumerate(trimmed_tag1_dict):
        client_df = df.loc[df["Tag1"] == key]
        client_df_list.append(client_df)
        client_n_samples = len(client_df.index)
        split_idx = int(client_n_samples * 0.8)
        client_df = shuffle(client_df)
        client_train_df = client_df.iloc[0:split_idx-1]
        client_test_df = client_df.iloc[split_idx:]
        
        client_train_path = f"./train/train_{client_num}/train.csv"
        client_test_path = f"./train/train_{client_num}/test.csv"
        if not os.path.exists(f"./train/train_{client_num}/"):
            os.makedirs(f"./train/train_{client_num}/")
        client_train_df.to_csv(client_train_path, sep=',', index=False, encoding='utf-8')
        client_test_df.to_csv(client_test_path, sep=',', index=False, encoding='utf-8')
        client_num += 1


create_stack_overflow_questions_dataset("./train-sample.csv")