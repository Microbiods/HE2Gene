import sys
import glob
import PIL
import pickle
import pathlib
import torch
import torchvision
import random
import openslide
import collections
import torchstain
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from .augmentation import EightSymmetry


def grid_norm_transform(patient, image_size, batch_size, gene_filter, num_workers, aux_task = False):

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
       
        count_mean += counts.mean(dim=0) * counts.shape[0]
        count_std += counts.std(dim=0) * counts.shape[0]

    train_mean /= n
    train_std /= n
    count_mean /= n
    count_std /= n
    
    return train_mean, train_std, count_mean, count_std

def get_neighbors(coord_x, coord_y):
    neighbors = []
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for dx, dy in directions:
        neighbor_x = coord_x + dx
        neighbor_y = coord_y + dy
        neighbors.append((neighbor_x, neighbor_y))

    return np.vstack(neighbors)



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
            # Can specify patient (take all sections)
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

        with open(self.root + "/subtypes.pkl", "rb") as f: # patient: subtype
            self.subtype = pickle.load(f)
        
        with open(self.root + "/all_genes.txt", "r") as f:
            self.all_genes = [line.strip() for line in f.readlines()]

        with open(self.root + "/target_genes.txt", "r") as f:
            self.target_genes = [line.strip() for line in f.readlines()]
            self.target_index = [self.all_genes.index(g) for g in self.target_genes]

        # with open(self.root + "aux_genes.txt", "r") as f:
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


        num_nbrs = 8
        nbr_count = np.zeros((num_nbrs, len(count))) 
        nbr_tumor = np.zeros((num_nbrs, 1), dtype=bool) 
        nbr_pixel = np.zeros((num_nbrs, 2), dtype=np.int64) 
        nbr_coord = get_neighbors(coord[0], coord[1]) 
        nbr_mask = np.ones((num_nbrs, 1), dtype=bool) 
        
        for idx, pos in enumerate(nbr_coord): 
            nbr_npz = "{}/{}/{}/{}_{}_{}.npz".format(self.root, self.subtype[patient], patient, section, pos[0], pos[1])
            if pathlib.Path(nbr_npz).exists():
                nbr_npz = np.load(nbr_npz)
                nbr_count[idx] = nbr_npz['count']
                nbr_tumor[idx] = nbr_npz['tumor']
                nbr_pixel[idx] = nbr_npz['pixel']
            else:
                nbr_mask[idx] = False 

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


        nbr_image = []
        for idx in range(num_nbrs):
            if nbr_mask[idx]: 
                nbr_img = slide.read_region((nbr_pixel[idx][0] - self.window // 2, nbr_pixel[idx][1] - self.window // 2), 0, (self.window, self.window))
                nbr_image.append(nbr_img.convert("RGB"))
            else: 
                nbr_image.append(PIL.Image.new("RGB", (self.window, self.window), (0, 0, 0))) 
        

        if self.transform is not None:
            image = self.transform(image) 
            nbr_image = [self.transform(nbr_img) for nbr_img in nbr_image] 
            
            if self.average:
                small_image = self.transform(small_image) 
                big_image = self.transform(big_image) 
                grid_images = torch.stack([image, small_image, big_image], dim=0) 

                tta_images = self.mean_transform(raw_image) 

                raw_small_images = self.mean_transform(raw_small_image) 
                raw_big_images = self.mean_transform(raw_big_image) 
                grid_tta_images = torch.stack([tta_images, raw_small_images, raw_big_images], dim=0)


        nbr_image = torch.stack(nbr_image, dim=0) 
       
        count = torch.as_tensor(count, dtype=torch.float) 
        nbr_count = torch.as_tensor(nbr_count, dtype=torch.float)
        
        if self.gene_norm == "log1p": 
            count = torch.log1p(1e4 * count / count.sum())
            nbr_count = torch.log1p(1e4 * nbr_count / nbr_count.sum(dim=1, keepdim=True))
            # will get nan if count.sum() == 0

        target_count = count[self.target_index]
        aux_count = count[self.aux_index]

        nbr_target_count = nbr_count[:, self.target_index]
        nbr_aux_count = nbr_count[:, self.aux_index]

        tumor = torch.as_tensor([1 if tumor else 0])
        nbr_tumor = torch.as_tensor(nbr_tumor, dtype=torch.int)

        coord = torch.as_tensor(coord)
        nbr_coord = torch.as_tensor(nbr_coord, dtype=torch.int)

        if self.average:
            return image, grid_images, tta_images, grid_tta_images, target_count, aux_count, nbr_image, nbr_target_count, nbr_aux_count, coord, nbr_coord, tumor, nbr_tumor, nbr_mask, pixel, patient, section # 33 x 35
        
        else:
            return image, target_count, aux_count, nbr_image, nbr_target_count, nbr_aux_count, coord, nbr_coord, tumor, nbr_tumor, nbr_mask, pixel, patient, section # 33 x 35

    # https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/34/11/10.1093_bioinformatics_bty030/3/bioinformatics_34_11_1966_s1.pdf?Expires=1683227533&Signature=Zlwblc0x~UGdpVId88RmD6HEA0Zdt~foTfxIjjgO4alNoYkGc4UB3QuGB3WG1OV269rjyq2H31mtaYgy6Vqt3qcGu8v~zuXo5VhFcZGZjaaBIv6I5NKuBwgoT4jW3exmzXzYJcHXL00xxVDCreKoemiQvM7EZGWgBMOhY4LlyVVanfp3-tZ0jL9nvXmGxq7zVdgjeUk0kyDV7Dl2LneV5Pj1ejFVNLLK2DRyOfYpIzJjziZ88fHkiPEfGic~GMJktEJ1XXNieOCcnW0-E1avLc-Mxbi3UTbieJoBHOUDZk2zVE-PAt2KWRP2QwCieC07BxFsL1fkiqiAt2iLktIhjw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
    # Spatial spots with typical number of cells in different regions of a breast cancer sample. (1) Ductal cancer in situ (~40 cells), (2) invasive cancer (~35 cells), (3) immune cell region (~200 cells) and (4) fibrous tissue (~5 cells).


image_mean = torch.tensor([0.5319, 0.5080, 0.6824])
image_std = torch.tensor([0.2442, 0.2063, 0.1633])

class SlideDataset(Dataset):
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
        
        image = PIL.Image.open(self.images[index]).convert("RGB")

        coord = np.array([int(i) for i in name.split("_")[-2:]])
        nbr_coord = get_neighbors(coord[0], coord[1]) 
        nbr_name = ["{}_{}_{}".format(name.split("_")[0], pos[0], pos[1]) for pos in nbr_coord]
        # only keep exists
        nbr_name = [name for name in nbr_name if pathlib.Path("{}/{}.tif".format(self.image_root, name)).exists()]

        nbr_image = []
        if len(nbr_name) != 0:
            for nbr in nbr_name:
                    nbr_image.append(PIL.Image.open(self.images[index]).convert("RGB"))
        else:
            nbr_image.append(image)
        
        if self.transform is not None:

            if self.average:
                image = self.mean_transform(image) # 8
                nbr_image = [self.mean_transform(nbr_img) for nbr_img in nbr_image]
            else:
                image = self.transform(image)
                nbr_image = [self.transform(nbr_img) for nbr_img in nbr_image]
            
        nbr_image = torch.stack(nbr_image, dim=0) 
        return name, image, nbr_image












