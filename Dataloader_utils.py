def parse_dataset_csv(csv_path):
    import csv
    import numpy as np
    dataset_list = []
    with open(csv_path) as csv_file:
        dataset_info = csv.reader(csv_file)
        col_name_list = []
        col_name_found = False
        for row in dataset_info:
            # The first row is the col names.
            if (not col_name_found):
                col_name_found = True
                col_name_list = row
                continue
            image_dic = {}
            image_dic['imagePath'] = row[0].replace('CheXpert-v1.0-small', '')
            image_dic['sex'] = row[1]
            image_dic['age'] = row[2]
            image_dic['direction'] = row[3]  # Frontal vs Lateral
            image_dic['type'] = row[4]  # PA(preferred) vs AP
            labels = np.array(row[5:19])
            # replace empty entry with 0
            labels[labels == ''] = '0.0'
            # replace -1 entry with 0
            labels[labels == '-1.0'] = '0.0'
            labels = labels.astype(np.float)
            image_dic['label'] = labels
            dataset_list.append(image_dic)
    return col_name_list, dataset_list


def select_subset(data_set, subset_size, seed=None):
    import random
    random.seed(seed)
    dataset_size = len(data_set)
    population = range(0, dataset_size - 1)
    chosen_idx = set()
    no_finding_class = 0
    # Keep adding cases until we have enough
    while (True):
        remaining_size = subset_size - len(chosen_idx)
        if remaining_size <= 0:
            # We have enough cases
            break
        idx_list = random.choices(population, k=remaining_size)
        for idx in idx_list:
            # Let's focus on the frontal PA scan for now.
            if data_set[idx]['direction'] == 'Frontal' and data_set[idx]['type'] == 'PA':
                if data_set[idx]['label'][0] > 0.5:
                    if no_finding_class >= subset_size / 14:
                        # We have too many 'no finding' classes
                        continue
                    no_finding_class = no_finding_class + 1
                chosen_idx.add(idx)
    chosen_idx = list(chosen_idx)
    return [data_set[idx] for idx in chosen_idx]


def save_dataset_info():
    import Constants
    import pickle
    _, full_train_dataset_info = parse_dataset_csv(Constants.TrainCSVpath)
    sub_train_dataset_info = select_subset(full_train_dataset_info, Constants.sizeOfsubset, Constants.seed)
    validation_dataset_info = parse_dataset_csv(Constants.ValidCSVpath)
    with open(Constants.ParsedDatasetPath, 'wb') as file_handler:
        pickle.dump(full_train_dataset_info, file_handler)
    with open(Constants.ParsedSubsetPath, 'wb') as file_handler:
        pickle.dump(sub_train_dataset_info, file_handler)
    with open(Constants.ParsedValidsetPath, 'wb') as file_handler:
        pickle.dump(validation_dataset_info, file_handler)


save_dataset_info()
