import numpy as np
import pandas as pd
import scanpy as sc
import torch
from itertools import compress
import argparse
import torchio as tio
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from collections import Counter
from models import Custom3DCNN, PatchEmbeddings
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from resnet.model import generate_model
from itertools import combinations


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, use_common_ids=True, phase='train', test_modality = None):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        # number of available modalities
        self.mn = np.array(data_dict['modality_nums'])
        self.observed = observed_idx
        ids = list(ids)
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        
        self.data_new = {modality: [data[i] for i in ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = [self.labels[i] for i in ids] #self.labels[ids]
        self.mc_new = self.mc[ids]
        self.mn_new = self.mn[ids]
        self.observed_new = self.observed[ids]
        self.data_root = '/home/data/ADNI/adni_miss/'
        self.combination_to_index = get_modality_combinations('TMFP')
        self.phase = phase

        #print("self.phase: ", self.phase)
        if self.phase=='test':
            modalities_list = ['T', 'M', 'F', 'P']
            full_name = ['Text', 'MRI','FDG', "AV45"]
            self.test_modality = test_modality # e.g. "TMFP"
            modality_mask = [1 if modality in self.test_modality else 0 for modality in modalities_list]
            modality_com_index = self.combination_to_index[self.test_modality]
            
            for i in range(len(modality_mask)):
                if modality_mask[i] == 0:
                    self.data_new[full_name[i]] == [-2] * len(self.data_new[full_name[i]])
            self.mc_new = np.array([modality_com_index] * len(self.mc_new))
            self.mn_new = np.array([sum(modality_mask)] * len(self.mn_new))
            self.observed_new = np.array([modality_mask] * len(self.observed_new))

        if self.phase in ["val"]:

            # Expand validation set by decomposing each sample into all non-empty
            # subsets of its available modalities
            modalities = ['T', 'M', 'F', 'P']
            new_ids = []
            new_label_new = []
            new_mc_new = []
            new_mn_new = []
            new_observed_new = []
            new_data_new = {modality: [] for modality in self.data_new}

            for i in range(len(self.ids)):
                observed_mask = list(self.observed_new[i])
                available_modalities = list(compress(modalities, observed_mask))
                if len(available_modalities) == 0:
                    continue
                for r in range(1, len(available_modalities) + 1):
                    for subset in combinations(available_modalities, r):
                        comb_str = ''.join(sorted(subset))
                        mask = [1 if m in subset else 0 for m in modalities]

                        new_ids.append(self.ids[i])
                        new_label_new.append(self.label_new[i])
                        new_mc_new.append(self.combination_to_index[comb_str])
                        new_mn_new.append(sum(mask))
                        new_observed_new.append(mask)
                        for modality_key in new_data_new.keys():
                            new_data_new[modality_key].append(self.data_new[modality_key][i])
            self.ids = new_ids
            self.label_new = new_label_new
            self.mc_new = np.array(new_mc_new)
            self.mn_new = np.array(new_mn_new)
            self.observed_new = np.array(new_observed_new)
            self.data_new = new_data_new
        # Sort ids by the number of available modalities
        #self.sorted_ids = list(sorted(np.arange(len(ids)), key=lambda x: self.mn_new[x], reverse=True))
        #self.data_new = {modality: [data[i] for i in self.sorted_ids] for modality, data in self.data_new.items() if 'modality' not in modality}
        #self.label_new =  [self.label_new[i] for i in self.sorted_ids]#self.label_new[self.sorted_ids]
        #print(Counter(self.label_new))
        #self.mc_new = self.mc_new[self.sorted_ids]
        #self.mn_new = self.mn_new[self.sorted_ids]
        #self.observed_new = self.observed_new[self.sorted_ids]

        self.label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
        
        # Image augmentation
        self.Spatial_transform = tio.OneOf({tio.RandomFlip(axes=0, flip_probability=0.5):0.33,
                                    tio.RandomAffine(scales=(0.9, 1.2), degrees=10, p=0.5):0.33,
                                    tio.RandomElasticDeformation(num_control_points=(10, 10, 10), max_displacement=8,locked_borders=2, p=0.5):0.33})
                                    #tio.RandomSwap(patch_size=10, num_iterations=20, p=0.5):0.25})


    def get_modality_combinations(self, modalities):
        all_combinations = []
        for i in range(len(modalities), 0, -1):
            comb = list(combinations(modalities, i))
            all_combinations.extend(comb)
        
        # Create a mapping dictionary
        combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
        #print(combination_to_index)
        return combination_to_index
    
    def list_to_combination(self, observed, modalities):
        # Convert observed list to a combination string (e.g., [1, 1, 1, 0] -> "TMF")
        return ''.join(compress(modalities, observed))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        label = self.label_new[idx]
        label = self.label_mapping[label]
        
        observed = self.observed_new[idx]  # Observed list, e.g., [1, 1, 1, 0]
        ori_observed = self.observed_new[idx] 
        modalities = ['T', 'M', 'F', 'P']  # Define modalities

        # Step 1: Identify available modalities
        available_modalities = list(compress(modalities, observed))  # E.g., ['T', 'M', 'F'] if observed = [1, 1, 1, 0]
        m = len(available_modalities)  # Number of available modalities

        # Step 2: Perform modality dropout augmentation
        if self.phase == 'train' and m > 1:  # Only apply during training and if more than 1 modality is available
            k = random.randint(1, m - 1)  # Randomly choose how many modalities to drop (1 <= k < m)
            modalities_to_drop = random.sample(available_modalities, k)  # Randomly select `k` modalities to drop
            #print(f"Dropping modalities: {modalities_to_drop}")  # Debugging output

            # Update the observed list to reflect dropped modalities
            for modality in modalities_to_drop:
                observed[modalities.index(modality)] = 0

        # Step 3: Compute the modality combination index (`mc`) based on the updated `observed` list
        combination = self.list_to_combination(observed, modalities)  # Convert observed to combination, e.g., "TM"
        combination = ''.join(sorted(combination))  # Sort the combination to ensure consistency
        mc = self.combination_to_index[combination]  # Get the index of the modality combination

        sample_data = {}
        modalities_full = ['Text', 'MRI', 'FDG', 'AV45']  # Define modalities
        for modality, data in self.data_new.items():
            if modality in ['MRI', 'FDG', 'AV45']:
                mask = ori_observed[modalities_full.index(modality)]
                sub_name = data[idx]
                if mask ==1:
                    #print('loading data..........')
                    Image_path = os.path.join(self.data_root, modality, sub_name+'.nii.gz')
                    img = nib.load(Image_path)
                    img = img.get_fdata()
                    [z, y, x] = img.shape
                    img = np.reshape(img, [1, z, y, x])
                    sample_data[modality] = img
                else:
                    #print('missing data..........')
                    sample_data[modality] = np.zeros((1, 128, 128, 128))
            else:
                text_data = data[idx]
                mask = ori_observed[0]
                if mask ==1:
                    sample_data[modality] = np.array(text_data).reshape(-1, 1)
                else:
                    sample_data[modality] = np.zeros((8, 1))

        if self.phase == 'train':
   
            MRI = sample_data['MRI']
            FDG = sample_data['FDG']
            AV45 = sample_data['AV45']
            #MRI = self.rescale(MRI)           #FDG = self.rescale(FDG)
            #AV45 = self.rescale(AV45)
            subject = tio.Subject(
                mri=tio.ScalarImage(tensor=MRI),
                fdg=tio.ScalarImage(tensor=FDG),
                av45=tio.ScalarImage(tensor=AV45)
            )
            subject = self.Spatial_transform(subject)
            #subject = self.Intensity_transform(subject)
            sample_data['MRI'] = subject.mri.numpy()
            sample_data['FDG'] = subject.fdg.numpy()
            sample_data['AV45'] = subject.av45.numpy()
            #sample_data = self.rescale(sample_data)


        return sample_data, label, mc, observed, ori_observed


def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_data(args, modality_dict):
    # Paths

    non_img_path = '/home/data/ADNI/adni/Table/ADNI_TABLE_TOTAL_normalized.csv'
    
    train_data = pd.read_excel('/home/data/ADNI/adni_miss/division/training_set.xlsx') # training file saves the Patient ID and boolean values of the existence of T1, FDG, and AV45 modalities
    train_data = train_data.sort_values(by='nums_mod', ascending=False, inplace=False)

    val_data = pd.read_excel('/home/data/ADNI/adni_miss/division/validation_set.xlsx') #validation file

    val_data = val_data.sort_values(by='nums_mod', ascending=False, inplace=False)
    test_data = pd.read_excel('/home/data/ADNI/adni_miss/division/testing_new.xlsx')  #testing file
    test_data = test_data.sort_values(by='nums_mod', ascending=False, inplace=False)

    all_data = pd.concat([train_data, val_data, test_data])

    exist_T1 = all_data['T1'].values.tolist()
    exist_FDG = all_data['PET-FDG'].values.tolist()
    exist_PET = all_data['PET'].values.tolist()
    subjects = all_data['Subject'].values.tolist()
    groups = all_data['Group'].values.tolist()
    all_data = all_data.set_index('Subject')
    nums_mods = all_data['nums_mod'].values.tolist()
    labels = all_data['DX'].values.tolist()
    n_labels = 3

    length_train = len(train_data['Subject'])
    length_val = len(val_data['Subject'])
    length_test = len(test_data['Subject'])

    train_idxs = list(range(length_train))
    valid_idxs = list(range(length_train, length_train + length_val))
    test_idxs = list(range(length_train + length_val, length_train + length_val + length_test))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}

    combination_to_index = get_modality_combinations(args.modality)

    mod_com = []
    nums_mods_new = []
    observed_idx_arr = []
    data_dict['Text'] = [-2]*len(exist_T1)
    data_dict['MRI'] = [-2]*len(exist_T1)
    data_dict['FDG'] = [-2]*len(exist_T1)
    data_dict['AV45'] = [-2]*len(exist_T1)
    
    def check_row_existence(df, subject, group):
        condition = (df["PTID"] == subject) & (df["COLPROT"] == group) & (df["VISM_IN"] == 0)
        return condition.any()
    
    non_image_data = pd.read_csv(non_img_path)
    non_image_data.set_index(['PTID', 'COLPROT', 'VISM_IN'], inplace=True)
    row_index = non_image_data.index
    for i in range(len(exist_T1)):
        com = ''
        if (subjects[i], groups[i], 0) in row_index:#check_row_existence(non_image_data, subjects[i], groups[i]):
            observed = [1]
            specific_data = non_image_data.loc[(subjects[i], groups[i], 0)]
            data = [specific_data['PTGENDER'], 
                    specific_data['PTEDUCAT'], 
                    specific_data['AGE'], 
                    specific_data['PTMARRY'], 
                    specific_data['APOE4'], 
                    specific_data['MMSE'], 
                    specific_data['ADNI_EF']
                    ,specific_data['ADNI_MEM']]
            #print(data)
            data_dict['Text'][i] = data
            com += 'T'
            nums_mods_new.append(1+nums_mods[i])
            
        else:
            nums_mods_new.append(nums_mods[i])
            observed = [0]
        observed.append(exist_T1[i])
        observed.append(exist_FDG[i])
        observed.append(exist_PET[i])
        if exist_T1[i] == 1:
            com += 'M'
            data_dict['MRI'][i] = subjects[i]
        if exist_FDG[i] == 1:
            com += 'F'
            data_dict['FDG'][i] = subjects[i]
        if exist_PET[i] == 1:
            com += 'P'
            data_dict['AV45'][i] = subjects[i]
        com = ''.join(sorted(set(com)))
        mod_com.append(combination_to_index[com])
        observed_idx_arr.append(observed)
        #print(com, '  :  ',combination_to_index[com],'  :   ', observed)
    
    print(combination_to_index)
    
    train_idxs_new = []
    valid_idxs_new = []
    test_idxs_new = []

    observed_idx_arr = np.array(observed_idx_arr)
    print('The number of training set: ', len(train_idxs))
    print('The number of validation set: ', len(valid_idxs))
    print('The number of test set: ', len(test_idxs))

    data_dict['modality_comb'] = mod_com

    id_to_idx = {id: idx for idx, id in enumerate(all_data.index)}
    common_idx_list = []
    #observed_idx_arr = np.zeros((labels.shape[0],4), dtype=bool) # IGCB order
    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)
    encoder_dict['MRI'] = torch.nn.DataParallel(torch.nn.Sequential(
        generate_model(18).cuda(),
        PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).cuda())
        )
    transforms['MRI'] = Compose([
                                ToTensor()
                            ])
    
    input_dims['MRI'] = args.hidden_dim

    encoder_dict['FDG'] = torch.nn.DataParallel(torch.nn.Sequential(
        generate_model(18).cuda(),
        PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).cuda()
        ))

    transforms['FDG'] = Compose([
                                ToTensor()
                            ])
    input_dims['FDG'] = args.hidden_dim

    encoder_dict['AV45'] = torch.nn.DataParallel(torch.nn.Sequential(
        generate_model(18).cuda(),
        PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).cuda()
        ))

    transforms['AV45'] = Compose([
                                ToTensor()
                            ])
    
    input_dims['AV45'] = args.hidden_dim
    encoder_dict['Text'] = torch.nn.DataParallel(nn.Linear(1, args.hidden_dim).cuda())

    transforms['Text'] = None
    input_dims['Text'] = args.hidden_dim
    data_dict['modality_nums'] = nums_mods_new

    #data_dict['MRI'] = [-2]*len(exist_T1)
    #data_dict['FDG'] = [-2]*len(exist_T1)
    #data_dict['AV45'] = [-2]*len(exist_T1)

    # data_dict save the a dictionary with five key words: ['mod1', 'mod2', 'mod3', 'mod4', 'modality_comb]
    # data_dict['modality_comb'] save a list with index of modality combination [0,2,5,7,4,3] by a index dictionary {'BCGI':0, "CGI": 2}
    # encoder_dict save the encoder of each modality
    # labels is a list saving the labels
    # train_idxs is a list saving the index of training data
    # valid_idxs is a list saving the index of validation data
    # test_idxs is a list saving the index of test data
    # n_labels save the number of labels
    # input_dims is a dictionary saving the dimension of each modality
    # transform is a dictionary saving the transform of each modality
    # masks is a dictionary saving the mask of each modality
    # full_modality_index equals to 0
    full_modality_index = 0
    # observed_idx_arr saving a boolean list with index of observed data, such as [[0,0,1,1],[0,1,1,0],[0,1,1,0]]
    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, 0, observed_idx_arr, full_modality_index

def collate_fn(batch):
    data, labels, mcs, observeds, ori_observeds = zip(*batch)
    modalities = data[0].keys()

    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    ori_observeds = torch.tensor(np.vstack(ori_observeds))
    return collated_data, labels, mcs, observeds, ori_observeds


def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, use_common_ids=True):

    train_transfrom = val_transform = test_transform = transforms
    #mask = masks['image']
    mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, use_common_ids, phase='train', test_modality = None)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, use_common_ids, phase='val', test_modality = None)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loaders = {}
    modalities = ['T', 'M', 'F', 'P']
    all_combinations = []
    for i in range(1, len(modalities) + 1):  # From 1 modality to all modalities
        all_combinations.extend(combinations(modalities, i))
    for combination in all_combinations:
        combination_str = ''.join(sorted(combination))
        test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, use_common_ids, phase='test', test_modality = combination_str )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
        test_loaders[combination_str] = test_loader

    return train_loader, train_loader_shuffle, val_loader, test_loaders



# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    #print(combination_to_index)
    return combination_to_index


def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--modality', type=str, default='TMFP') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=16) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=4) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=3)

    return parser.parse_known_args()

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

if __name__ == '__main__':
    modalities  = 'TMFP'
    args, _ = parse_args() 
    modality_dict = {'Text':0, 'MRI': 1, 'FDG': 2, 'AV45': 3}
    data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index  = load_and_preprocess_data(args, modality_dict)
    train_loader, train_loader_shuffle, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.use_common_ids)
    i = 0
    print('Validation set: ', len(val_loader))

    



