import Constants


def parse_dataset_csv(csv_path):
    import csv
    import numpy as np
    dataset_list = []
    key_flag = 'label'
    num_of_lateral = 0
    num_of_case = 0
    with open(csv_path) as csv_file:
        dataset_info = csv.reader(csv_file)
        col_name_list = []
        col_name_found = False
        previous_record = None
        current_record = None
        image_dic = {}
        no_finding_class = 0
        for row in dataset_info:
            # The first row is the col names.
            if not col_name_found:
                col_name_found = True
                col_name_list = row
                continue
            path_list = row[0].split('/')
            previous_record = current_record
            current_record = path_list[2]+path_list[2] # patient#+study#
            folder_name = row[0].split('/')[0]
            if current_record == previous_record:  # The same patient in the same study,
                if row[3] == 'Lateral':
                    image_dic['Lateral_imagePath'] = Constants.DatasetRootDir+row[0].replace(folder_name, '')
                    num_of_lateral = num_of_lateral + 1
                else:
                    image_dic['Frontal_imagePath'] = Constants.DatasetRootDir + row[0].replace(folder_name, '')
                    image_dic['type'] = row[4]  # PA(preferred) vs AP
            else:
                if key_flag in image_dic:
                    dataset_list.append(image_dic) # Write the old record before init a new one
                image_dic = {}
                image_dic['Lateral_imagePath'] = None
                image_dic['Frontal_imagePath'] = None
                image_dic['type'] = None
                image_dic['name'] = current_record
                if row[3] == 'Lateral':
                    image_dic['Lateral_imagePath'] = Constants.DatasetRootDir+row[0].replace(Constants.dir_prefix, '')
                else:
                    image_dic['Frontal_imagePath'] = Constants.DatasetRootDir + row[0].replace(Constants.dir_prefix, '')
                    image_dic['type'] = row[4]  # PA(preferred) vs AP
                image_dic['sex'] = row[1]
                image_dic['age'] = row[2]
                labels = np.array(row[5:19])
                # replace empty entry with 0
                labels[labels == ''] = '0.0'
                # replace -1 entry with 0
                labels[labels == '-1.0'] = '0.0'
                labels = labels.astype(np.float)
                if labels[0] > 0.5:
                    labels[1:len(labels)] = 0.0
                    no_finding_class = no_finding_class + 1
                    if no_finding_class <= Constants.num_of_nofinding_cases:
                        image_dic['label'] = np.squeeze(labels)
                        num_of_case = num_of_case + 1
                else:
                    image_dic['label'] = np.squeeze(labels)
                    num_of_case = num_of_case + 1
        print("Num of cases with lateral image: "+str(num_of_lateral))
        print("Total num of cases: " + str(num_of_case))
        print("Total num of no finding cases: " + str(no_finding_class))
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
            if data_set[idx]['Lateral_imagePath'] is not None and data_set[idx]['type'] == 'PA':
                if data_set[idx]['label'][0] > 0.5:
                    if no_finding_class >= subset_size / 14:
                        # We have too many 'no finding' classes
                        continue
                    no_finding_class = no_finding_class + 1
                chosen_idx.add(idx)
    chosen_idx = list(chosen_idx)
    return [data_set[idx] for idx in chosen_idx]


def select_concat(data_set):
    chosen_idx = set()
    for idx in range(len(data_set)):
        # Let's focus on the frontal PA scan for now.
        if data_set[idx]['Lateral_imagePath'] is not None and data_set[idx]['type'] == 'PA':
            chosen_idx.add(idx)
    chosen_idx = list(chosen_idx)
    return [data_set[idx] for idx in chosen_idx]


def save_dataset_info():
    import Constants
    import pickle
    _, full_train_dataset_info = parse_dataset_csv(Constants.TrainCSVpath)
    sub_train_dataset_info = select_subset(full_train_dataset_info, Constants.sizeOfsubset, Constants.seed)
    validation_dataset_info = parse_dataset_csv(Constants.ValidCSVpath)
    full_train_dataset_info = select_concat(full_train_dataset_info)
    with open(Constants.ParsedDatasetPath_wide, 'wb') as file_handler:
        pickle.dump(full_train_dataset_info, file_handler)
    with open(Constants.ParsedSubsetPath_wide, 'wb') as file_handler:
        pickle.dump(sub_train_dataset_info, file_handler)
    with open(Constants.ParsedValidsetPath_wide, 'wb') as file_handler:
        pickle.dump(validation_dataset_info, file_handler)


save_dataset_info()
