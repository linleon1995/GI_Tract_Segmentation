import os
import pandas as pd
import numpy as np
from data.data_utils import get_files, get_cv_split


# TODO: sample balance
def build_dataset(data_path, fold, num_fold=5):
    assert fold < num_fold, 'The fold is not allowed'
    split_path = '../split'
    case_id_path = os.path.join(split_path, f'cv-{num_fold}')
    cases = get_files(os.path.join(data_path, 'train'), recursive=False, get_dirs=True, return_fullpath=False)

    # get split
    if not os.path.isdir(case_id_path) or len(os.listdir(case_id_path))<num_fold:
        os.makedirs(case_id_path, exist_ok=True)
        cv_indices = get_cv_split(num_fold, len(cases), shuffle=True)
        for fold in cv_indices:
            split_cases = np.take(cases, cv_indices[fold]['test'])
            split_cases = split_cases[0][...,None]
            df = pd.DataFrame(split_cases)
            df.to_csv(os.path.join(case_id_path, f'{fold}_fold.csv'), header=None, index=False)

    fold_files = get_files(case_id_path, 'csv')
    train_data_info, valid_data_info = [], []
    test_df = pd.read_csv(
        os.path.join(case_id_path, f'{fold}_fold.csv'), header=None)
    test_fold = test_df[0].tolist()
    train_fold = list(set(cases)-set(test_fold))
    num_train = len(train_fold)
    train_fold, valid_fold = train_fold[:int(num_train*0.9)], train_fold[int(num_train*0.9):]

    # TODO: this process is pretty slow, consider store it
    # TODO: remove no foreground cases?
    # get training table
    data_info = pd.read_csv(os.path.join(data_path, 'train.csv'))
    for case in train_fold:
        train_data_info.append(data_info.loc[data_info['id'].str.contains(case)])
    train_data_info = pd.concat(train_data_info)
    
    for case in valid_fold:
        valid_data_info.append(data_info.loc[data_info['id'].str.contains(case)])
    valid_data_info = pd.concat(valid_data_info)

    train_data_info = 3

    # rle decoding to get segmentation
    # get image path


def eda(root):
    images = get_files(root, 'png')
    cases = get_files(root, recursive=False, get_dirs=True)

    total_days = []
    for case in cases:
        days = len(get_files(case, recursive=False, get_dirs=True))
        total_days.append(days)

    print(30*'-', 'EDA', 30*'-')
    print(f'Case number: {len(cases)}')
    print(f'Image number: {len(images)}')
    print(f'[max] {max(total_days)} day of single case')
    print(f'[min] {min(total_days)} day of single case')
    print(f'[mean] {sum(total_days)/len(total_days)} day of single case')


if __name__ == '__main__':
    root = rf'C:\Users\test\Desktop\Leon\Datasets\UW_Madison'
    # eda(root)

    fold = 0
    build_dataset(root, fold, num_fold=5)