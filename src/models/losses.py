import torch
import torch.nn as nn



class SimpMSELoss(nn.Module):
    def __init__(self, num_genes):
        super(SimpMSELoss, self).__init__()
        self.num_genes = num_genes

    def forward(self, x_pred, x):
    
        loss = torch.sum((x_pred - x) ** 2) / self.num_genes

        return loss



class SpatMSELoss(nn.Module):
    def __init__(self, num_genes):
        super(SpatMSELoss, self).__init__()
        self.num_genes = num_genes

    def forward(self, x, x_neighbors, x_labels, x_neighbors_labels, mask, num_genes):
       
        bs, num_neighbors, features = x_neighbors.shape
        # device = x.device
        x_expanded = x.unsqueeze(1).repeat(1, 8, 1) # (256, 250) -> (256, 8, 250)
        x_labels_expanded = x_labels.unsqueeze(1).repeat(1, 8, 1) # (256, 1) -> (256, 8, 1)

        same_label_mask = (x_labels_expanded == x_neighbors_labels)  # (256, 8, 1)
        same_label_valid_mask = (same_label_mask & mask).squeeze(1) # (256, 8, 1)

        x_expanded = x_expanded.view(-1, features) # (256, 8, 250) -> (256*8, 250)
        x_neighbors = x_neighbors.view(-1, features) # (256, 8, 250) -> (256*8, 250)
        same_label_valid_mask = same_label_valid_mask.view(-1) # (256, 8, 1) -> (256*8, 1)

        # if same_label_valid_mask.sum() == 0:
        #     return torch.tensor(0.0).to(device)
        # else:

        loss = torch.sum((x_expanded[same_label_valid_mask] - x_neighbors[same_label_valid_mask]) ** 2) / self.num_genes

        return loss


class SpatialMSELoss(nn.Module):
    def __init__(self):
        super(SpatialMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, x_nbr, tumor, nbr_tumor, mask):
       
        # if torch.isnan(x_nbr).any():
        #     print('nan in nbr_mean')

        # num_genes = x_nbr.shape[2]
        # device = x.device

        # x = x.unsqueeze(1).repeat(1, 2, 1) # (bs, 250) -> (bs, 8, 250)
        tumor = tumor.unsqueeze(1).repeat(1, 8, 1) # (bs, 1) -> (bs, 8, 1)

        mask = (tumor == nbr_tumor) & mask # (bs, 8, 1)


        denom = torch.sum(mask, dim=1) # (bs, 1)
        nbr_sum = torch.sum(x_nbr * mask, dim=1) # (bs, 250)

        # print(denom)

        if (denom>0).sum() == 0:
            nbr_mean = x.clone()

        else:
            nonzero_mask = (denom>0).squeeze()
            x = x[nonzero_mask]
            nbr_mean = nbr_sum[nonzero_mask]/denom[nonzero_mask]

        # if x.shape != nbr_mean.shape:
        #     print('shapes not equal')

        return self.mse_loss(x, nbr_mean)



import torch
import torch.nn as nn

class CVWeightedMSELoss(nn.Module):
    def __init__(self, cv_weights):
        super().__init__()
        self.cv_weights = cv_weights  # weights for each output dimension

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true)**2, dim=0)  # calculate MSE for each output dimension
        cv = torch.std(y_true, dim=0) / torch.mean(y_true, dim=0)  # calculate CV for each output dimension
        cv_weight = 1 / (cv + 1e-6)  # calculate weight for each output dimension
        cv_weight *= self.cv_weights  # multiply by user-defined weights
        
        return torch.mean(cv_weight * mse)  # apply weighted MSE and return the mean loss







# class AnnotateMSELoss(nn.Module):
#     def __init__(self):
#         super(AnnotateMSELoss, self).__init__()
#         self.bce_loss = nn.MSELoss()

#     def forward(self, predictions, targets, mask):
       
#         bs, num_neighbors, features = predictions.shape
#         # device = predictions.device

#         predictions = predictions.view(-1, features) # (256, 8, 250) -> (256*8, 250)
#         targets = targets.view(-1, features) # (256, 8, 250) -> (256*8, 250)
#         mask = mask.view(-1) # (256, 8, 1) -> (256*8, 1)

#         # if mask.sum() == 0:
#         #     return torch.tensor(0.0).to(device)
#         # else:
#         return self.bce_loss(predictions[mask], targets[mask])








# class SpatialMSELoss(nn.Module):
#     def __init__(self):
#         super(SpatialMSELoss, self).__init__()
#         self.mse_loss = nn.MSELoss(reduction='none')

#     def forward(self, x, x_neighbors, x_labels, x_neighbors_labels, mask):
#         # x: (batch_size, features) -> (256, 250)
#         # x_neighbors: (batch_size, num_neighbors, features) -> (256, 8, 250)

#         # Expand x to have the same size as x_neighbors
#         x_expanded = x.unsqueeze(1).repeat(1, 8, 1)
#         # x_expanded: (batch_size, num_neighbors, features) -> (256, 8, 250)
#         x_labels_expanded = x_labels.unsqueeze(1).repeat(1, 8, 1)


#         # Find neighbors with the same label as the corresponding element in x
#         same_label_mask = (x_labels_expanded == x_neighbors_labels)
#         # same_label_mask: (batch_size, num_neighbors, 1) -> (256, 8, 1)

#         same_label_neighbors = x_neighbors * same_label_mask.float()
#         # same_label_neighbors: (batch_size, num_neighbors, features) -> (256, 8, 250)

#         # Calculate the MSE loss between x and x_neighbors
#         loss_per_sample = self.mse_loss(x_expanded, same_label_neighbors)
#         # loss_per_sample: (batch_size, num_neighbors, features) -> (256, 8, 250)

#         # Apply the mask to the loss, so that only valid neighbors contribute to the loss
#         masked_loss_per_sample = loss_per_sample * mask.float()
#         # masked_loss_per_sample: (batch_size, num_neighbors, features) -> (256, 8, 250)

#         # Calculate the final loss, averaging over all samples and valid neighbors
#         loss = torch.sum(masked_loss_per_sample) / torch.sum(mask.float()) / torch.sum(mask.float())
#         # loss: scalar

#         return loss


# class SpatialMSELoss(nn.Module):
#     def __init__(self):
#         super(SpatialMSELoss, self).__init__()
#         self.mse_loss = nn.MSELoss()

#     def forward(self, x, x_neighbors, x_labels, x_neighbors_labels, mask):
       
#         bs, num_neighbors, features = x_neighbors.shape
#         device = x.device
#         x_expanded = x.unsqueeze(1).repeat(1, num_neighbors, 1) # (256, 250) -> (256, 8, 250)
#         x_labels_expanded = x_labels.unsqueeze(1).repeat(1, num_neighbors, 1) # (256, 1) -> (256, 8, 1)

#         loss_all_batch = []
#         for i in range(bs):
#             same_label_mask = (x_labels_expanded[i] == x_neighbors_labels[i]) 
#             same_label_valid_mask = (same_label_mask & mask[i]).squeeze(1) 

#             if x_expanded[i][same_label_valid_mask].shape[0] != 0: # in case there is no valid neighbor

#                 loss_per_batch = self.mse_loss(x_expanded[i][same_label_valid_mask], x_neighbors[i][same_label_valid_mask]) 
            
#                 loss_all_batch.append(loss_per_batch)

#         if len(loss_all_batch) == 0: # in case there is no valid neighbor
#             return torch.tensor(0.0).to(device)
#         else:
#             return torch.mean(torch.stack(loss_all_batch))



# class SpatialTumorBCELoss(nn.Module):
#     def __init__(self):
#         super(SpatialTumorBCELoss, self).__init__()
#         self.bce_loss = nn.MSELoss()

#     def forward(self, predictions, targets, mask):
       
#         bs, num_neighbors, features = predictions.shape
#         device = predictions.device

#         loss_all_batch = []
#         for i in range(bs):
#             valid_mask = mask[i].squeeze(1) 
#             if predictions[i][valid_mask].shape[0] != 0:
#                 loss_per_batch = self.bce_loss(predictions[i][valid_mask], targets[i][valid_mask]) 
#                 loss_all_batch.append(loss_per_batch)

#         if len(loss_all_batch) == 0: # in case there is no valid neighbor
#             return torch.tensor(0.0).to(device)
#         else:
#             return torch.mean(torch.stack(loss_all_batch))








# # Create random data for demonstration
# x = torch.randn(256, 250)
# x_neighbors = torch.randn(256, 8, 250)
# x_labels = torch.randint(0, 2, (256, 1))
# x_neighbors_labels = torch.randint(0, 2, (256, 8, 1))
# x_neighbors_labels_pred = torch.randint(256, 8, 1)
# mask = torch.randint(0, 2, (256, 8, 1)).to(torch.bool)

# # Instantiate the custom loss class
# custom_loss = SpatialMSELoss()
# mask_bce_loss = MaskedBCELoss()
# # Calculate the loss
# # loss = custom_loss(x, x_neighbors, x_labels, x_neighbors_labels, mask)
# loss = mask_bce_loss(x_neighbors_labels, x_neighbors_labels_pred, mask)
# print(loss)

# # Calculate the gradient
# loss.backward()
