import os

import pandas as pd
from data.data_utils import get_files, get_cv_split


def build_dataset(cv_path, fold, num_fold=5):
    assert fold < num_fold, 'fold not allowed'
    case_id_path = os.path.join(cv_path, f'cv-{num_fold}')
    if not os.path.isdir(case_id_path):
        # images = get_files(root, 'png')
        cases = get_files(root, recursive=False, get_dirs=True)
        get_cv_split(num_fold, len(cases), save_path=case_id_path)
        # save_fold_cases(num_fold=num_fold, num_sample=len(cases))

    fold_files = get_files(case_id_path, 'csv')
    train_fold_files, valid_fold_files = [], []
    for fold_file in fold_files:
        df = pd.read_csv(os.path.join(case_id_path, f'{fold}_fold.csv'), header=None)
        if f'{fold}_fold' in fold_file:
            valid_fold_files.append(fold_file)
        else:
            train_fold_files.append(fold_file)



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
    root = rf'C:\Users\test\Desktop\Leon\Datasets\UW_Madison\train'
    # eda(root)

    fold = 0
    build_dataset(root, fold, num_fold=5)