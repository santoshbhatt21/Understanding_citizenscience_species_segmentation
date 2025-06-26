import torch
state_dict = torch.load("E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/best_model_140_0.09.pth", map_location='cpu')
print(state_dict['classifier.1.weight'].shape)