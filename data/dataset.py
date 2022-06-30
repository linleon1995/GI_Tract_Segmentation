import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data.data_utils import get_files, get_cv_split
from utils.rle import semantic_rle_decode


# TODO: sample balance
# TODO: HU value?
def build_dataset(data_path, fold, num_fold):
    train_seg_path = os.path.join('..\split', f'cv-{num_fold}', 'segment', f'{fold}_train.csv')
    valid_seg_path = os.path.join('..\split', f'cv-{num_fold}', 'segment', f'{fold}_valid.csv')
    if os.path.exists(train_seg_path) and os.path.exists(valid_seg_path):
        train_seg = pd.read_csv(train_seg_path)
        valid_seg = pd.read_csv(valid_seg_path)
    else:
        train_seg, valid_seg = build_segmentation(data_path, fold, num_fold=5)
        train_seg.reset_index(inplace=True)
        valid_seg.reset_index(inplace=True)

    train_dataset = GIDataset(data_path, train_seg)
    valid_dataset = GIDataset(data_path, valid_seg)
    for x in train_dataset:
        print(x)
    # x = next(iter(train_dataset))
    print(train_seg['id'])
    # print(train_seg.shape)
    print(3)


class GIDataset(Dataset):
    def __init__(self, data_root, data_info, remove_empty=True):
        self.data_root = data_root
        if remove_empty:
            data_info = data_info
        self.data_info = data_info
        
    def __len__(self):
        return self.data_info.shape[0]
    
    def __getitem__(self, idx):
        sample_id = self.data_info.iloc[idx]['id']
        case, day, _, slice = sample_id.split('_')
        sample_df = self.data_info[self.data_info['id']==sample_id]
        
        # get image path
        img_root= os.path.join(self.data_root, 'train', case, f'{case}_{day}', 'scans')
        img_path = get_files(img_root, f'slice_{slice}')[0]
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        # img2 = plt.imread(img_path)
        
        # rle decoding to get segmentation
        if sample_df['segmentation'].isnull().values.all():
            mask = np.zeros_like(img)
        else:
            total_rle_code = {class_idx: rle_code \
                for class_idx, rle_code in enumerate(sample_df['segmentation'], 1) \
                if rle_code is not np.nan}
            mask = semantic_rle_decode(total_rle_code, shape=img.shape)
            # plt.imshow(img, 'gray')
            # plt.imshow(mask, alpha=0.1)
            # plt.show()
        
        # print(3)
        return {'input':img, 'target': mask}
        # return {'input':raw_chunk, 'target': target, 'tmh_name': tmh_name}


def build_segmentation(data_path, fold, num_fold=5):
    assert fold < num_fold, 'The fold is not allowed'
    split_path = '..\split'
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
    train_fold.sort()
    num_train = len(train_fold)
    train_fold, valid_fold = train_fold[:int(num_train*0.9)], train_fold[int(num_train*0.9):]

    # TODO: this process is pretty slow, consider store it

    # get training table
    seg_dir = os.path.join(case_id_path, 'segment')    
    os.makedirs(seg_dir, exist_ok=True)

    train_path = os.path.join(seg_dir, f'{fold}_train.csv')
    if not os.path.exists(train_path):
        data_info = pd.read_csv(os.path.join(data_path, 'train.csv'))
        for case in train_fold:
            train_data_info.append(data_info.loc[data_info['id'].str.contains(case)])
        train_data_info = pd.concat(train_data_info)
        train_data_info.to_csv(train_path, index=False)
    else:
        print(f'{train_path} is already exists.')
    
    valid_path = os.path.join(seg_dir, f'{fold}_valid.csv')
    if not os.path.exists(valid_path):
        for case in valid_fold:
            valid_data_info.append(data_info.loc[data_info['id'].str.contains(case)])
        valid_data_info = pd.concat(valid_data_info)
        valid_data_info.to_csv(valid_path, index=False)
    else:
        print(f'{valid_path} is already exists.')

    return train_data_info, valid_data_info


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