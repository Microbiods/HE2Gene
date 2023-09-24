import torch
import torch.nn as nn
from torchvision.models import resnet50

# resnet50(weights="IMAGENET1K_V2") # IMAGENET1K_V1
# resnet50(weights=None)

class ResNet50(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(ResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, num_target_genes)

    def forward(self, x):
        x = self.base_model(x)
        output = self.fc(x)
        return output
    


class ResNet50Extractor(nn.Module):
    def __init__(self, pretrain = False):
        super(ResNet50Extractor, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        x = self.base_model(x)
        return x

class AuxResNet50(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False):
        super(AuxResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, num_target_genes)
        self.fc2 = nn.Linear(2048, num_aux_genes)

    def forward(self, x):
        x = self.base_model(x)
        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2
    

class AnnotateResNet50(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(AnnotateResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, num_target_genes)
        self.fc2 = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        output1 = self.fc1(x)
        output2 = torch.sigmoid(self.fc2(x))
        return output1, output2


class AuxAnnotateResNet50(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False, extraction = False):
        super(AuxAnnotateResNet50, self).__init__()
        self.extraction = extraction

        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, num_target_genes)
        self.fc2 = nn.Linear(2048, num_aux_genes)
        self.fc3 = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        if self.extraction:
            return x
        else:
            output1 = self.fc1(x)
            output2 = self.fc2(x)
            output3 = torch.sigmoid(self.fc3(x))
            return output1, output2, output3
    
class AuxAnnotateResNet50HER2(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False, extraction = False):
        super(AuxAnnotateResNet50HER2, self).__init__()
        self.extraction = extraction

        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, num_target_genes)
        self.fc2 = nn.Linear(2048, num_aux_genes)
        self.fc3 = nn.Linear(2048, 7)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        if self.extraction:
            return x
        else:
            output1 = self.fc1(x)
            output2 = self.fc2(x)
            output3 = self.fc3(x)
            return output1, output2, output3
    





from torchvision.models import resnet50, vgg16, inception_v3, densenet121, efficientnet_b0, convnext_tiny, vit_b_16, swin_t
class AuxAnnotateDeepNets(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False, architecture = None):
        super(AuxAnnotateDeepNets, self).__init__()
        if pretrain:
            weight = "IMAGENET1K_V1"
        else:
            weight = None

        self.architecture = architecture

        if self.architecture == 'vgg':
            self.base_model = vgg16(weights=weight) # weights=None
            self.base_model.classifier[6] = nn.Identity()  # Remove the final fully connected layer
            output_dim = 4096 

        elif self.architecture == 'inception':
            self.base_model = inception_v3(weights=weight)
            self.base_model.fc = nn.Identity() 
            output_dim = 2048 
        
        elif self.architecture == 'resnet':
            self.base_model = resnet50(weights=weight)
            self.base_model.fc = nn.Identity() 
            output_dim = 2048 

        elif self.architecture == 'densenet':
            self.base_model = densenet121(weights=weight)
            self.base_model.classifier = nn.Identity() 
            output_dim = 1024 

        elif self.architecture == 'efficientnet':
            self.base_model = efficientnet_b0(weights=weight)
            self.base_model.classifier[1] = nn.Identity()  
            output_dim = 1280  

        elif self.architecture == 'convnext':
            self.base_model = convnext_tiny(weights=weight)
            self.base_model.classifier[2] = nn.Identity()  
            output_dim = 768 

        elif self.architecture == 'vit':
            self.base_model = vit_b_16(weights=weight)
            self.base_model.heads[0] = nn.Identity() 
            output_dim = 768 

        elif self.architecture == 'swint':
            self.base_model = swin_t(weights=weight)
            self.base_model.head = nn.Identity()  
            output_dim = 768 

        self.fc1 = nn.Linear(output_dim, num_target_genes)
        self.fc2 = nn.Linear(output_dim, num_aux_genes)
        self.fc3 = nn.Linear(output_dim, 1)  # Binary classification head

    def forward(self, x):

        if  self.architecture == 'inception' and self.base_model.training: # train mode
            x, _ = self.base_model(x)
        else:
            x = self.base_model(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        output3 = torch.sigmoid(self.fc3(x))
        return output1, output2, output3

class SpatialResNet50(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(SpatialResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, num_target_genes)

    def forward(self, x, x_nbr): # x_neighbor is the neighbor spots, cost more memory

        bs, channels, height, width = x.shape
        num_nbr = x_nbr.shape[1]
        x_nbr = x_nbr.reshape(-1, channels, height, width)

        x_combined = torch.cat((x, x_nbr), dim=0)
        x_combined = self.base_model(x_combined)

        x = x_combined[:bs]
        x_nbr = x_combined[bs:]

        # TODO: can calculate the similarity between x and x_nbr here
        
        x_nbr = x_nbr.reshape(bs, num_nbr, -1)

        output = self.fc(x)  # for target genes
        output_nbr = self.fc(x_nbr)  # only predict target genes for neighbor spots, not aux genes

        return output, output_nbr



class AnnotateNet(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(AnnotateNet, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        output = torch.sigmoid(self.fc(x))
        return output




class AuxAnnotateSpatialResNet50(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False, extraction = False):
        super(AuxAnnotateSpatialResNet50, self).__init__()
        self.extraction = extraction
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, num_target_genes)
        self.fc_aux = nn.Linear(2048, num_aux_genes)
        self.fc_tmr = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x, x_nbr): # x_neighbor is the neighbor spots, cost more memory
        
        if self.training:
            bs, channels, height, width = x.shape
            num_nbr = x_nbr.shape[1]
            x_nbr = x_nbr.reshape(-1, channels, height, width)

            x_combined = torch.cat((x, x_nbr), dim=0)
            x_combined = self.base_model(x_combined)

            x = x_combined[:bs]
            x_nbr = x_combined[bs:]

            # TODO: can calculate the similarity between x and x_nbr here
            
            x_nbr = x_nbr.reshape(bs, num_nbr, -1)

            output = self.fc(x)  # for target genes
            output_nbr = self.fc(x_nbr)  # only predict target genes for neighbor spots, not aux genes

            target_output = self.fc_aux(x)  # for aux genes
            target_output_nbr = self.fc_aux(x_nbr)

            tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation
            tmr_output_nbr = torch.sigmoid(self.fc_tmr(x_nbr))

            return output, target_output, tmr_output, output_nbr, target_output_nbr, tmr_output_nbr
        
        else:
            x = self.base_model(x)
            if self.extraction:
                return x
            else:
                output = self.fc(x)  # for target genes
                target_output = self.fc_aux(x)  # for aux genes
                tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation

                return output, target_output, tmr_output




class FixedAuxAnnotateSpatialResNet50(nn.Module):
    def __init__(self, num_target_genes, num_aux_genes, pretrain = False):
        super(FixedAuxAnnotateSpatialResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, num_target_genes)
        self.fc_aux = nn.Linear(2048, num_aux_genes)
        self.fc_tmr = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x, x_nbr): # x_neighbor is the neighbor spots, cost more memory
        
        if self.training:
            bs, channels, height, width = x.shape
            x = self.base_model(x)
            output = self.fc(x)  # for target genes
            target_output = self.fc_aux(x)  # for aux genes
            tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation
            
            with torch.no_grad():
                num_nbr = x_nbr.shape[1]
                x_nbr = x_nbr.reshape(-1, channels, height, width)
                x_nbr = self.base_model(x_nbr)
                x_nbr = x_nbr.reshape(bs, num_nbr, -1)
                output_nbr = self.fc(x_nbr)  # only predict target genes for neighbor spots, not aux genes
                target_output_nbr = self.fc_aux(x_nbr)
                tmr_output_nbr = torch.sigmoid(self.fc_tmr(x_nbr))

            return output, target_output, tmr_output, output_nbr, target_output_nbr, tmr_output_nbr
        
        else:
            x = self.base_model(x)
            output = self.fc(x)  # for target genes
            target_output = self.fc_aux(x)  # for aux genes
            tmr_output = torch.sigmoid(self.fc_tmr(x))  # for spatial annotation

            return output, target_output, tmr_output









class TumorResNet50(nn.Module):
    def __init__(self, pretrain = False):
        super(TumorResNet50, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, 1)  # Binary classification head

    def forward(self, x):
        x = self.base_model(x)
        output1 = torch.sigmoid(self.fc1(x))
        return output1



class TumorResNet50Extractor(torch.nn.Module):
    def __init__(self, pretrain=False):
        super(TumorResNet50Extractor, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = torch.nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        x = self.base_model(x)
        return x



class HE2RNA(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(HE2RNA, self).__init__()
        if pretrain:
            self.base_model = resnet50(weights="IMAGENET1K_V1")
        else:
            self.base_model = resnet50(weights=None)
        self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_target_genes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output



class STNet(nn.Module):
    def __init__(self, num_target_genes, pretrain = False):
        super(STNet, self).__init__()
        if pretrain:
            self.base_model = densenet121(weights="IMAGENET1K_V1")
        else:
            self.base_model = densenet121(weights=None)
        self.base_model.classifier = nn.Identity() 
        self.fc = nn.Linear(1024, num_target_genes)

    def forward(self, x):
        x = self.base_model(x)
        output = self.fc(x)
        return output

if __name__ == "__main__":
    
    model = AuxAnnotateSpatialResNet50(250, 10000)
    # test the model use random input
    x = torch.randn(2, 3, 224, 224)
    x_neighbor = torch.randn(2, 8, 3, 224, 224)
    x_neighbor_mask = torch.randn(2, 1)
    output1, output2, output3, output1_neighbor, output2_neighbor, output3_neighbor = model(x, x_neighbor)
    print(output1.shape)
    