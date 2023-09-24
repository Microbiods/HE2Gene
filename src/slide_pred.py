import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.gridloader import SlideDataset
from utils.normalize import SetSeed
from models.resnet import AuxAnnotateSpatialResNet50
import pandas as pd


if __name__ == "__main__":

    # pass a parameter from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/exps', help='root directory of patches')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--average', type=bool, default=True, help='average')
    args = parser.parse_args()

    # load txt
    with open('./src/target_genes.txt', 'r') as f:
        target_genes = f.read().splitlines()
    with open('./src/aux_genes.txt', 'r') as f:
        aux_genes = f.read().splitlines()

    SetSeed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # window size: 150
    image_mean = torch.tensor([0.5319, 0.5080, 0.6824])
    image_std = torch.tensor([0.2442, 0.2063, 0.1633])
    
    # define image transformations
    test_transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),    
                transforms.Normalize(mean=image_mean, std=image_std)
    ])

    # test dataset
    test_dataset = SlideDataset(image_root=args.root,
                                transform=test_transform,
                                average=args.average)

    test_dataloader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=args.num_workers, 
                                    pin_memory=True)
    

    # for name, img, nbr_image in test_dataloader:
    #     print(name)
    #     print(img.shape, nbr_image.shape)
    #     # break

    num_target_genes = 250
    num_aux_genes = 19699
    model = AuxAnnotateSpatialResNet50(num_target_genes, num_aux_genes, pretrain = True)

    model_path = './checkpoint/model_spat.pt'

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    with torch.no_grad():
        all_name = []
        all_count_pred = []
        all_aux_count_pred = []
        all_tumor_pred = []
        for step, (name, image, nbr_image) in enumerate(test_dataloader):
            image = image.to(device)
            nbr_image = nbr_image.to(device)
            if args.average:
                count_pred, aux_count_pred, tumor_pred = model(image[0], nbr_image)
                count_pred = count_pred.mean(0).unsqueeze(0)
                aux_count_pred = aux_count_pred.mean(0).unsqueeze(0)
                tumor_pred = tumor_pred.mean(0).unsqueeze(0)
            else:
                count_pred, aux_count_pred, tumor_pred = model(image)
            
            all_name.append(name[0])
            all_count_pred.append(count_pred.cpu().numpy())
            all_aux_count_pred.append(aux_count_pred.cpu().numpy())
            all_tumor_pred.append(tumor_pred.cpu().numpy()>0.5)

        all_count_pred = np.concatenate(all_count_pred)
        all_aux_count_pred = np.concatenate(all_aux_count_pred)
        all_tumor_pred = np.concatenate(all_tumor_pred)

        # generate dataframe
        all_count_pred = pd.DataFrame(all_count_pred, columns=target_genes, index=all_name)
        all_aux_count_pred = pd.DataFrame(all_aux_count_pred, columns=aux_genes, index=all_name)
        all_tumor_pred = pd.DataFrame(all_tumor_pred, columns=['Annotation'], index=all_name)

        # save to output
        all_count_pred.to_csv('./output/spat/target_count_pred.csv')
        all_aux_count_pred.to_csv('./output/spat/aux_count_pred.csv')
        all_tumor_pred.to_csv('./output/spat/tumor_pred.csv')
    
        

