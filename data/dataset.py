import os

from data.data_utils import get_files, get_cv_split


# def build_dataset():
#     if not os.path.isdir()
#         cases = get_files(root, 'png')
#         save_fold_cases(num_fold=5, num_sample=len(cases))
#     else:



# def save_fold_cases(num_fold, num_sample, shuffle=True, seed=0):
#     # TODO: sample blanace
#     cv_indices = get_cv_split(num_fold, num_sample, shuffle, seed)


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
    eda(root)