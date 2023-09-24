import sys
import glob
import PIL
import pickle
import random
import pathlib
import torch
import torchvision
import openslide
import collections
import torchstain
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
# from graph_construction import calcADJ
import scanpy as sc
from .augmentation import EightSymmetry


def norm_transform(patient, image_size, batch_size, gene_filter, num_workers, aux_task = False):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), 
        transforms.ToTensor()         
    ])

    train_dataset = BreastCancerSTDataset(patient=patient, 
                                        transform=transform,
                                        gene_filter=gene_filter,
                                        aux_task = aux_task
                                        )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_mean = 0
    train_std = 0
    count_mean = 0
    count_std = 0
    n = 0
    for (images, counts, *_) in train_dataloader:
        n += images.shape[0]
        train_mean += images.mean(dim=(0, 2, 3)) * images.shape[0]
        train_std += images.std(dim=(0, 2, 3)) * images.shape[0]
       
        # calculate mean and std of counts
        count_mean += counts.mean(dim=0) * counts.shape[0]
        count_std += counts.std(dim=0) * counts.shape[0]

    train_mean /= n
    train_std /= n
    count_mean /= n
    count_std /= n
    
    return train_mean, train_std, count_mean, count_std


def crop_image(slide, spot, window):
    images = []
    for pix in spot:
        img = slide.read_region((pix[0] - window // 2, pix[1] - window // 2), 0, (window, window))
        img = img.convert("RGB")
        
        img = np.array(img)

        images.append(img)
    return images


class BreastCancerSTDataseHER2(Dataset):
    def __init__(self,
                 root=None, 
                 patient=None, 
                 window=None,
                 transform=None,
                 mean_transform=None,
                 gene_norm=None, 
                 subsample=None,
                 average=False,
                 ):

        self.dataset = glob.glob("{}/*/*/*.npz".format(root))
        
        if subsample:
            sample_size = int(len(self.dataset) * subsample)
            self.dataset = random.sample(self.dataset, sample_size)

        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-3] in patient))]
            self.npzs = [np.load(data)for data in self.dataset]

        else:
           raise ValueError()
        
        self.root = root
        self.window = window
        self.transform = transform
        self.mean_transform = mean_transform
        self.gene_norm = gene_norm
        self.average = average

        with open("./data/breast/all_genes.txt", "r") as f:
            self.all_genes = [line.strip() for line in f.readlines()]

        with open("./data/breast/overlap_genes.txt", "r") as f:
            self.target_genes = [line.strip() for line in f.readlines()]
            self.target_index = [self.all_genes.index(g) for g in self.target_genes]

        with open("./data/breast/aux_genes.txt", "r") as f:
        # with open(self.root + "/full_aux_genes.txt", "r") as f: # use all left genes for aux task
            self.aux_genes = [line.strip() for line in f.readlines()]
            self.aux_index = [self.all_genes.index(g) for g in self.aux_genes]


        self.slide = collections.defaultdict(dict)
        for (patient, section) in set([(d.split("/")[-3], d.split("/")[-2]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}/{}/{}/wsi.tif".format(self.root, patient, section))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = self.npzs[index]
        count   = npz["count"]
        tumor   = npz["tumor"]
        pixel   = npz["pixel"] 
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"] 

        slide = self.slide[patient][section]
        image = slide.read_region((int(pixel[0]) - self.window // 2, int(pixel[1]) - self.window // 2), 0, (self.window, self.window))
        image = image.convert("RGB")
          
        raw_image = copy.deepcopy(image)

        if self.average:
            # small_wingow = 225
            # big_window = 300
            small_wingow = 75
            big_window = 225
            slide = self.slide[patient][section]
            small_image = slide.read_region((int(pixel[0]) - small_wingow // 2, int(pixel[1]) - small_wingow // 2), 0, (small_wingow, small_wingow))
            small_image = small_image.convert("RGB")
            big_image = slide.read_region((int(pixel[0]) - big_window // 2, int(pixel[1]) - big_window // 2), 0, (big_window, big_window))
            big_image = big_image.convert("RGB")
            raw_small_image = copy.deepcopy(small_image)
            raw_big_image = copy.deepcopy(big_image)


        if self.transform is not None:
            image = self.transform(image) # 1

            if self.average:
                small_image = self.transform(small_image) # 1
                big_image = self.transform(big_image) # 1
                grid_images = torch.stack([image, small_image, big_image], dim=0) # 3

                tta_images = self.mean_transform(raw_image) # 8

                raw_small_images = self.mean_transform(raw_small_image) # 8
                raw_big_images = self.mean_transform(raw_big_image) # 8
                grid_tta_images = torch.stack([tta_images, raw_small_images, raw_big_images], dim=0) # 3 x 8

        count = torch.as_tensor(count, dtype=torch.float)

        if self.gene_norm == "log1p": 
            count = torch.log1p(1e4 * count / count.sum())
        elif self.gene_norm == "raw": 
            count = torch.log1p(count)

        target_count = count[self.target_index]
        aux_count = count[self.aux_index]
        tumor = torch.as_tensor(tumor)
        coord = torch.as_tensor(coord)

        if self.average:
            return image, grid_images, tta_images, grid_tta_images, target_count, aux_count, coord, tumor, pixel, patient, section # 33 x 35
        else:
            return image, target_count, aux_count, coord, tumor, pixel, patient, section # 33 x 35

    # https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/34/11/10.1093_bioinformatics_bty030/3/bioinformatics_34_11_1966_s1.pdf?Expires=1683227533&Signature=Zlwblc0x~UGdpVId88RmD6HEA0Zdt~foTfxIjjgO4alNoYkGc4UB3QuGB3WG1OV269rjyq2H31mtaYgy6Vqt3qcGu8v~zuXo5VhFcZGZjaaBIv6I5NKuBwgoT4jW3exmzXzYJcHXL00xxVDCreKoemiQvM7EZGWgBMOhY4LlyVVanfp3-tZ0jL9nvXmGxq7zVdgjeUk0kyDV7Dl2LneV5Pj1ejFVNLLK2DRyOfYpIzJjziZ88fHkiPEfGic~GMJktEJ1XXNieOCcnW0-E1avLc-Mxbi3UTbieJoBHOUDZk2zVE-PAt2KWRP2QwCieC07BxFsL1fkiqiAt2iLktIhjw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
    # Spatial spots with typical number of cells in different regions of a breast cancer sample. (1) Ductal cancer in situ (~40 cells), (2) invasive cancer (~35 cells), (3) immune cell region (~200 cells) and (4) fibrous tissue (~5 cells).



class BreastCancerSTDataset(Dataset):
    def __init__(self,
                 root=None,  
                 patient=None,
                 window=None, 
                 transform=None,
                 mean_transform=None,
                 gene_norm=None, 
                 subsample=None,
                 average=False,
                 ):

        self.dataset = glob.glob("{}/*/*/*.npz".format(root))
        
        if subsample:
            sample_size = int(len(self.dataset) * subsample)
            self.dataset = random.sample(self.dataset, sample_size)

        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient))]
            self.npzs = [np.load(data)for data in self.dataset]

        else:
           raise ValueError()
        
        self.root = root
        self.window = window
        self.transform = transform
        self.mean_transform = mean_transform
        self.gene_norm = gene_norm
        self.average = average

        with open(self.root + "/subtypes.pkl", "rb") as f: 
            self.subtype = pickle.load(f)

        with open(self.root + "/all_genes.txt", "r") as f:
            self.all_genes = [line.strip() for line in f.readlines()]

        with open(self.root + "/target_genes.txt", "r") as f:
            self.target_genes = [line.strip() for line in f.readlines()]
            self.target_index = [self.all_genes.index(g) for g in self.target_genes]

        # with open(self.root + "/aux_genes.txt", "r") as f:
        with open(self.root + "/full_aux_genes.txt", "r") as f: # use all left genes for aux task
            self.aux_genes = [line.strip() for line in f.readlines()]
            self.aux_index = [self.all_genes.index(g) for g in self.aux_genes]

        # load WSI
        self.slide = collections.defaultdict(dict)
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}/{}/{}/{}_{}_wsi.tif".format(self.root, self.subtype[patient], patient, patient, section))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        npz = self.npzs[index]
        count   = npz["count"]
        tumor   = npz["tumor"]
        pixel   = npz["pixel"] 
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"] 
       
        slide = self.slide[patient][section]
        image = slide.read_region((pixel[0] - self.window // 2, pixel[1] - self.window // 2), 0, (self.window, self.window))
        image = image.convert("RGB")
        raw_image = copy.deepcopy(image)

        if self.average:
            # small_wingow = 225
            # big_window = 300
            small_wingow = 75
            big_window = 225
            slide = self.slide[patient][section]
            small_image = slide.read_region((pixel[0] - small_wingow // 2, pixel[1] - small_wingow // 2), 0, (small_wingow, small_wingow))
            small_image = small_image.convert("RGB")
            big_image = slide.read_region((pixel[0] - big_window // 2, pixel[1] - big_window // 2), 0, (big_window, big_window))
            big_image = big_image.convert("RGB")
            raw_small_image = copy.deepcopy(small_image)
            raw_big_image = copy.deepcopy(big_image)


        if self.transform is not None:
            image = self.transform(image) # 1

            if self.average:
                small_image = self.transform(small_image) # 1
                big_image = self.transform(big_image) # 1
                grid_images = torch.stack([image, small_image, big_image], dim=0) # 3

                tta_images = self.mean_transform(raw_image) # 8

                raw_small_images = self.mean_transform(raw_small_image) # 8
                raw_big_images = self.mean_transform(raw_big_image) # 8
                grid_tta_images = torch.stack([tta_images, raw_small_images, raw_big_images], dim=0) # 3 x 8

        count = torch.as_tensor(count, dtype=torch.float)

        if self.gene_norm == "log1p": 
            count = torch.log1p(1e4 * count / count.sum())
        elif self.gene_norm == "raw": 
            count = torch.log1p(count)

        target_count = count[self.target_index]
        aux_count = count[self.aux_index]
        tumor = torch.as_tensor([1 if tumor else 0])
        coord = torch.as_tensor(coord)

        if self.average:
            return image, grid_images, tta_images, grid_tta_images, target_count, aux_count, coord, tumor, pixel, patient, section # 33 x 35
        else:
            return image, target_count, aux_count, coord, tumor, pixel, patient, section # 33 x 35

    # https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/34/11/10.1093_bioinformatics_bty030/3/bioinformatics_34_11_1966_s1.pdf?Expires=1683227533&Signature=Zlwblc0x~UGdpVId88RmD6HEA0Zdt~foTfxIjjgO4alNoYkGc4UB3QuGB3WG1OV269rjyq2H31mtaYgy6Vqt3qcGu8v~zuXo5VhFcZGZjaaBIv6I5NKuBwgoT4jW3exmzXzYJcHXL00xxVDCreKoemiQvM7EZGWgBMOhY4LlyVVanfp3-tZ0jL9nvXmGxq7zVdgjeUk0kyDV7Dl2LneV5Pj1ejFVNLLK2DRyOfYpIzJjziZ88fHkiPEfGic~GMJktEJ1XXNieOCcnW0-E1avLc-Mxbi3UTbieJoBHOUDZk2zVE-PAt2KWRP2QwCieC07BxFsL1fkiqiAt2iLktIhjw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
    # Spatial spots with typical number of cells in different regions of a breast cancer sample. (1) Ductal cancer in situ (~40 cells), (2) invasive cancer (~35 cells), (3) immune cell region (~200 cells) and (4) fibrous tissue (~5 cells).


image_mean = torch.tensor([0.5319, 0.5080, 0.6824])
image_std = torch.tensor([0.2442, 0.2063, 0.1633])

class PatchDataset(Dataset):
    def __init__(self,
                 image_root=None,
                 transform=None,
                 average=False,
                 ):
    
        self.image_root = image_root
        self.images = glob.glob("{}/*.tif".format(image_root))

        self.transform = transform
        self.average = average

        self.mean_transform = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            EightSymmetry(),
                                            transforms.Lambda(lambda symmetries: torch.stack([transforms.Normalize(mean=image_mean, std=image_std)(transforms.ToTensor()(s)) for s in symmetries]))])


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        name = self.images[index].split("/")[-1].split(".")[0]
        
        image = PIL.Image.open(self.images[index])
        image = image.convert("RGB")

        if self.transform is not None:

            if self.average:
                image = self.mean_transform(image) # 8
            else:
                image = self.transform(image)

        return name, image 



# test dataloader
if __name__ == "__main__":

    root = '.data/breast/' 

    with open(root + "subtypes.pkl", "rb") as f:
        subtypes = pickle.load(f)
    patient = list(subtypes.keys())
    
    params = {
    "image_size": 224,
    "batch_size": 256,
    "gene_filter": None,
    "num_workers": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "patience": 10,
    "warmup_epochs": 5,
    "min_learning_rate": 1e-5,
    }
    
    for test_patient in patient:
        train_patient = [p for p in patient if p != test_patient]
        print(f'Test patient: {test_patient}')

        image_mean, image_std, count_mean, count_std = norm_transform(
                                                        train_patient, 
                                                        params['image_size'], 
                                                        params['batch_size'], 
                                                        gene_filter = params['gene_filter'],
                                                        num_workers = params['num_workers'],
                                                        aux_task = True
                                                        )

        train_transform = transforms.Compose([
            transforms.Resize((params['image_size'], params['image_size'])), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))]),
            transforms.ToTensor(),          
            transforms.Normalize(mean=image_mean, std=image_std) 
        ])

      
        train_dataset = BreastCancerSTDataset(patient=train_patient, transform=train_transform, gene_filter = params['gene_filter'], aux_task = True)
        train_len = int(len(train_dataset) * 0.9)
        val_len = len(train_dataset) - train_len
        train_data, val_data = random_split(train_dataset, [train_len, val_len])
        train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=True)
        val_dataloader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=True)

        test_transform = transforms.Compose([
                    transforms.Resize((params['image_size'], params['image_size'])), # resize image to the specified size
                    transforms.ToTensor(),          # convert image to a PyTorch tensor
                    transforms.Normalize(mean=image_mean, std=image_std) # normalize image using the train mean and std
                ])

        # test dataset
        test_dataset = BreastCancerSTDataset(patient=[test_patient], transform=test_transform, gene_filter = params['gene_filter'])
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=True)
        
        zero_indices1 = torch.nonzero(count_mean == 0)
        zero_indices2 = torch.nonzero(count_std == 0)
        print(zero_indices1, zero_indices2)

        for (image, count, coord, pixel, tumor, patient, section) in tqdm(train_dataloader):
            count = (count - count_mean) / count_std
            print()
        
       

    # 10x Genomics Visium
    # Slide-seq
    # seqFISH
    # MERFISH
    # STARmap