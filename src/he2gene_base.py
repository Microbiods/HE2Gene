import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import copy
import argparse
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import torchvision.transforms as transforms
from datasets.dataloader import BreastCancerSTDataset
from utils.normalize import SetSeed
from models.resnet import AuxAnnotateResNet50
import os
import time
from datasets.augmentation import EightSymmetry
from torch.optim.lr_scheduler import StepLR


def test(test_dataloader, model, criterion):
    model.eval()
    with torch.no_grad():
        test_single_loss = 0
        test_grid_loss = 0
        test_tta_loss = 0
        test_grid_tta_loss = 0

        real_counts = []
        single_pred_counts = []
        grid_pred_counts = []
        tta_pred_counts = []
        grid_tta_pred_counts = []

        real_aux_counts = []
        single_pred_aux_counts = []
        grid_pred_aux_counts = []
        tta_pred_aux_counts = []
        grid_tta_pred_aux_counts = []
        
        real_tumors = []
        single_pred_tumors = []
        grid_pred_tumors = []
        tta_pred_tumors = []
        grid_tta_pred_tumors = []

        pixels = []
        coordinates = []
        annotations = []
        patients = []
        sections = []

        for step, (image, grid_images, tta_images, grid_tta_images, count, aux_count, coord, tumor, pixel, patient, section) \
            in enumerate(test_dataloader):
            
            count = count.to(device)
            aux_count = aux_count.to(device)
            tumor = tumor.to(device).to(torch.float32)

            image = image.to(device) # 1
            grid_images = grid_images.to(device) # 3
            tta_images = tta_images.to(device) # 8
            grid_tta_images = grid_tta_images.to(device) # 24

            batch, c, h, w = image.shape
            num_grid = grid_images.shape[1]
            num_tta = tta_images.shape[1]
            grid_tta_images = grid_tta_images.reshape(batch, -1, c, h, w)
            num_grid_tta = grid_tta_images.shape[1]

            combined_images = torch.cat((image.unsqueeze(1), grid_images, tta_images, grid_tta_images), dim=1)
            image = combined_images.reshape(-1, c, h, w)
            count_pred, aux_count_pred, tumor_pred = model(image)

            count_pred = count_pred.reshape(batch, 1+num_grid+num_tta+num_grid_tta, -1)
            aux_count_pred = aux_count_pred.reshape(batch, 1+num_grid+num_tta+num_grid_tta, -1)
            tumor_pred = tumor_pred.reshape(batch, 1+num_grid+num_tta+num_grid_tta, -1)

            single_count_pred = count_pred[:, 0, :] # 1
            single_aux_count_pred = aux_count_pred[:, 0, :]
            single_tumor_pred = tumor_pred[:, 0, :]

            grid_count_pred = count_pred[:, 1:1+num_grid, :].mean(dim=1) # 3
            grid_aux_count_pred = aux_count_pred[:, 1:1+num_grid, :].mean(dim=1)
            grid_tumor_pred = tumor_pred[:, 1:1+num_grid, :].mean(dim=1)

            tta_count_pred = count_pred[:, 1+num_grid:1+num_grid+num_tta, :].mean(dim=1) # 8
            tta_aux_count_pred = aux_count_pred[:, 1+num_grid:1+num_grid+num_tta, :].mean(dim=1)
            tta_tumor_pred = tumor_pred[:, 1+num_grid:1+num_grid+num_tta, :].mean(dim=1)

            grid_tta_count_pred = count_pred[:, 1+num_grid+num_tta:, :].mean(dim=1)
            grid_tta_aux_count_pred = aux_count_pred[:, 1+num_grid+num_tta:, :].mean(dim=1)
            grid_tta_tumor_pred = tumor_pred[:, 1+num_grid+num_tta:, :].mean(dim=1)
                                        
            # get single loss
            single_loss = criterion(single_count_pred, count)
            test_single_loss += single_loss.item()

            # get grid loss
            grid_loss = criterion(grid_count_pred, count)
            test_grid_loss += grid_loss.item()

            # get tta loss
            tta_loss = criterion(tta_count_pred, count)
            test_tta_loss += tta_loss.item()

            # get grid tta loss
            grid_tta_loss = criterion(grid_tta_count_pred, count)
            test_grid_tta_loss += grid_tta_loss.item()

            real_counts.append(count.cpu().numpy())
            single_pred_counts.append(single_count_pred.cpu().detach().numpy())
            grid_pred_counts.append(grid_count_pred.cpu().detach().numpy())
            tta_pred_counts.append(tta_count_pred.cpu().detach().numpy())
            grid_tta_pred_counts.append(grid_tta_count_pred.cpu().detach().numpy())
            
            real_aux_counts.append(aux_count.cpu().numpy())
            single_pred_aux_counts.append(single_aux_count_pred.cpu().detach().numpy())
            grid_pred_aux_counts.append(grid_aux_count_pred.cpu().detach().numpy())
            tta_pred_aux_counts.append(tta_aux_count_pred.cpu().detach().numpy())
            grid_tta_pred_aux_counts.append(grid_tta_aux_count_pred.cpu().detach().numpy())
           
            real_tumors.append(tumor.cpu().numpy())
            single_pred_tumors.append(single_tumor_pred.cpu().detach().numpy())
            grid_pred_tumors.append(grid_tumor_pred.cpu().detach().numpy())
            tta_pred_tumors.append(tta_tumor_pred.cpu().detach().numpy())
            grid_tta_pred_tumors.append(grid_tta_tumor_pred.cpu().detach().numpy())
          
            pixels.append(pixel.numpy())
            coordinates.append(coord.numpy())
            annotations.append(tumor.cpu().numpy())
            patients.extend(patient)
            sections.extend(section)

    real_counts = np.concatenate(real_counts)
    single_pred_counts = np.concatenate(single_pred_counts)
    grid_pred_counts = np.concatenate(grid_pred_counts)
    tta_pred_counts = np.concatenate(tta_pred_counts)
    grid_tta_pred_counts = np.concatenate(grid_tta_pred_counts)

    real_aux_counts = np.concatenate(real_aux_counts)
    single_pred_aux_counts = np.concatenate(single_pred_aux_counts)
    grid_pred_aux_counts = np.concatenate(grid_pred_aux_counts)
    tta_pred_aux_counts = np.concatenate(tta_pred_aux_counts)
    grid_tta_pred_aux_counts = np.concatenate(grid_tta_pred_aux_counts)

    real_tumors = np.concatenate(real_tumors)
    single_pred_tumors = np.concatenate(single_pred_tumors)
    grid_pred_tumors = np.concatenate(grid_pred_tumors)
    tta_pred_tumors = np.concatenate(tta_pred_tumors)
    grid_tta_pred_tumors = np.concatenate(grid_tta_pred_tumors)

    pixels = np.concatenate(pixels)
    coordinates = np.concatenate(coordinates)
    annotations = np.concatenate(annotations)

    test_single_loss = test_single_loss/len(test_dataloader)
    test_grid_loss = test_grid_loss/len(test_dataloader)
    test_tta_loss = test_tta_loss/len(test_dataloader)
    test_grid_tta_loss = test_grid_tta_loss/len(test_dataloader)

    print(f'   Test Single Loss: {test_single_loss:.4f}, Grid Loss: {test_grid_loss:.4f}, TTA Loss: {test_tta_loss:.4f}, Grid TTA Loss: {test_grid_tta_loss:.4f}')

    return test_single_loss, test_grid_loss, test_tta_loss, test_grid_tta_loss, \
            real_counts, single_pred_counts, grid_pred_counts, tta_pred_counts, grid_tta_pred_counts, \
            real_aux_counts, single_pred_aux_counts, grid_pred_aux_counts, tta_pred_aux_counts, grid_tta_pred_aux_counts, \
            real_tumors, single_pred_tumors, grid_pred_tumors, tta_pred_tumors, grid_tta_pred_tumors, \
            pixels, coordinates, annotations, patients, sections



if __name__ == "__main__":

    # pass a parameter from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--aux', type=float, default=20.0, help='weight for aux loss')
    parser.add_argument('--tmr', type=float, default=1.0, help='weight for tmr loss')
    args = parser.parse_args()

    start_time = time.time()

    data_root = './data/breast/'
    save_root = './output/base'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(data_root + "/subtypes.pkl", "rb") as f:
        subtypes = pickle.load(f)
    all_patient = list(subtypes.keys())
    params = {
    "crop_size": 150,
    "image_size": 224,
    "batch_size": 256,   
    "num_workers": 8,
    "learning_rate": 1e-5,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 10,
    "subsample": False, # 0.1 
    "average": True
    }
    params_str = '; '.join([f"{key}: {value}" for key, value in params.items()])
    print('Parameters: ' + '{' + params_str + '}' + '\n')

    test_patient = ['BC23377', 'BC23268', 'BC23508', 'BC23903',  'BC23287']
    train_patient = [p for p in all_patient if p not in test_patient]

    ### get image mean and std

    # norm_transform = transforms.Compose([
    # transforms.Resize((params['image_size'], params['image_size'])), # crop first then resize
    # transforms.ToTensor()         
    # ])
    # train_dataset = BreastCancerSTDataset(data_root,
    #                                     patient=train_patient,
    #                                     window=params['crop_size'],
    #                                     transform=norm_transform,
    #                                     gene_norm='log1p',
    #                                     subsample=params['subsample'],
    #                                     )
    # train_dataloader = DataLoader(train_dataset, 
    #                                 batch_size=params['batch_size'], 
    #                                 shuffle=False, 
    #                                 num_workers=params['num_workers'],
    #                                 )
    # train_image = []
    # for (image, count, *_) in train_dataloader:
    #     train_image.append(image)
    # train_image = torch.cat(train_image, dim=0)
    # image_mean = train_image.mean(dim=(0, 2, 3)) 
    # image_std = train_image.std(dim=(0, 2, 3)) 
    # print(f'Image mean: {image_mean}, Image std: {image_std}')
    
    # window size: 150
    image_mean = torch.tensor([0.5319, 0.5080, 0.6824])
    image_std = torch.tensor([0.2442, 0.2063, 0.1633])
    
    # define image transformations
    train_transform = transforms.Compose([
                    transforms.Resize((params['image_size'], params['image_size'])), 
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomApply([transforms.RandomRotation((90, 90))]),
                    transforms.ToTensor(),         
                    transforms.Normalize(mean=image_mean, std=image_std)
                ])

    test_transform = transforms.Compose([
                transforms.Resize((params['image_size'], params['image_size'])),
                transforms.ToTensor(),    
                transforms.Normalize(mean=image_mean, std=image_std)
    ])

    # train dataset
    train_dataset = BreastCancerSTDataset(data_root,
                                        patient=train_patient,
                                        window=params['crop_size'],
                                        transform=train_transform,
                                        gene_norm='log1p',
                                        subsample=params['subsample'],
                                        )
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=params['batch_size'], 
                                    shuffle=True, 
                                    num_workers=params['num_workers'], 
                                    pin_memory=True)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset)-int(len(train_dataset)*0.9)])
                            
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=params['batch_size'], 
                                    shuffle=True, 
                                    num_workers=params['num_workers'], 
                                    pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                    batch_size=params['batch_size'],
                                    shuffle=False,
                                    num_workers=params['num_workers'],
                                    pin_memory=True)
                            
    if params['average']:
        mean_transform = transforms.Compose([
                                            transforms.Resize((params['image_size'], params['image_size'])),
                                            EightSymmetry(),
                                            transforms.Lambda(lambda symmetries: torch.stack([transforms.Normalize(mean=image_mean, std=image_std)(transforms.ToTensor()(s)) for s in symmetries]))])

        # test dataset
        test_dataset = BreastCancerSTDataset(data_root,
                                        patient=test_patient,
                                        window=params['crop_size'],
                                        transform=test_transform,
                                        mean_transform=mean_transform,
                                        gene_norm='log1p',
                                        subsample=params['subsample'],
                                        average=params['average']
                                        )
    else:
        test_dataset = BreastCancerSTDataset(data_root,
                                        patient=test_patient,
                                        window=params['crop_size'],
                                        transform=test_transform,
                                        gene_norm='log1p',
                                        subsample=params['subsample'],
                                        )
        


    test_dataloader = DataLoader(test_dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    num_workers=params['num_workers'], 
                                    pin_memory=True)
    
    num_target_genes = train_dataset[0][1].shape[0]
    num_aux_genes = train_dataset[0][2].shape[0]
    model = AuxAnnotateResNet50(num_target_genes, num_aux_genes, pretrain = True)
    model.to(device)

    criterion = nn.MSELoss()
    criterion_tmr = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 
                            lr=params['learning_rate'], 
                            weight_decay=params['weight_decay'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5) # decay the learning rate by a factor of 0.5 every 10 epochs


    best_val_loss = np.inf
    best_model = None
    epochs_no_improvement = 0

    # Train
    for epoch in range(params['epochs']):
        print(f'Epoch {epoch}:')

        model.train()
        train_traget_loss = 0
        # train_aux_loss = 0
        real_counts = []
        pred_counts = []
        # real_aux_counts = []
        # pred_aux_counts = []
        
        for step, (image, count, aux_count, coord, tumor, *_) in enumerate(train_dataloader):
                
            count = count.to(device)
            aux_count = aux_count.to(device)
            tumor = tumor.to(device).to(torch.float32)
            image = image.to(device)

            optimizer.zero_grad()
            count_pred, aux_count_pred, tmr_pred = model(image)
            traget_loss = criterion(count_pred, count)
            aux_loss = criterion(aux_count_pred, aux_count)
            tmr_loss = criterion_tmr(tmr_pred, tumor)
            
            loss = traget_loss + args.aux * aux_loss + args.tmr * tmr_loss 

            train_traget_loss += traget_loss.item()
            # train_aux_loss += aux_loss.item()

            real_counts.append(count.cpu().numpy())
            pred_counts.append(count_pred.cpu().detach().numpy())
            # real_aux_counts.append(aux_count.cpu().numpy())
            # pred_aux_counts.append(aux_count_pred.cpu().detach().numpy())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # if step % 50 == 0:
            #     print(f'  Step {step}: Loss: {loss.item():.4f}')

        real_counts = np.concatenate(real_counts)
        pred_counts = np.concatenate(pred_counts)
        # real_aux_counts = np.concatenate(real_aux_counts)
        # pred_aux_counts = np.concatenate(pred_aux_counts)

        train_traget_loss = train_traget_loss/len(train_dataloader)

        print(f'  Train Loss: {train_traget_loss:.4f}')
        
        # # Validation for each section
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for step, (image, count, aux_count, coord, tumor, *_) in enumerate(val_dataloader):
                count = count.to(device)
                image = image.to(device)
                count_pred, aux_count_pred, tmr_pred = model(image)
                loss = criterion(count_pred, count)
                val_loss += loss.item()

            val_loss = val_loss/len(val_dataloader)
            print(f'  Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
        if epochs_no_improvement >= params['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

        scheduler.step()

    # Test
    test_single_loss, test_grid_loss, test_tta_loss, test_grid_tta_loss, \
        real_counts, single_pred_counts, grid_pred_counts, tta_pred_counts, grid_tta_pred_counts, \
        real_aux_counts, single_pred_aux_counts, grid_pred_aux_counts, tta_pred_aux_counts, grid_tta_pred_aux_counts, \
        real_tumors, single_pred_tumors, grid_pred_tumors, tta_pred_tumors, grid_tta_pred_tumors, \
        pixels, coordinates, annotations, patients, sections = test(test_dataloader, best_model, criterion)

    # Save 
    os.makedirs(save_root, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(save_root, "model.pt"))

    np.savez_compressed(os.path.join(save_root, "predictions.npz"),
        real_counts=real_counts, 
        single_pred_counts=single_pred_counts,
        grid_pred_counts=grid_pred_counts,
        tta_pred_counts=tta_pred_counts,
        grid_tta_pred_counts=grid_tta_pred_counts,

        real_aux_counts=real_aux_counts,
        single_pred_aux_counts=single_pred_aux_counts,
        grid_pred_aux_counts=grid_pred_aux_counts,
        tta_pred_aux_counts=tta_pred_aux_counts,
        grid_tta_pred_aux_counts=grid_tta_pred_aux_counts,

        real_tumors=real_tumors,
        single_pred_tumors=single_pred_tumors,
        grid_pred_tumors=grid_pred_tumors,
        tta_pred_tumors=tta_pred_tumors,
        grid_tta_pred_tumors=grid_tta_pred_tumors,

        pixels=pixels,
        coordinates=coordinates,
        annotations=annotations,
        patients=patients, 
        sections=sections,
        traget_genes=test_dataset.target_genes,
        aux_genes=test_dataset.aux_genes,
        )

    end_time = time.time()
    print(f'Total time: {end_time - start_time:.4f} seconds')

    print()
