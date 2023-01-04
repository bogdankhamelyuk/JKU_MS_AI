from dataset import count_all_jsons
from dataset import DroneThrmImg_Dataset
from torch.utils.data import DataLoader

all_files = count_all_jsons()
train_files = int(0.8*all_files)
test_files = int(0.2*all_files)

test_dataset = DroneThrmImg_Dataset(req_samples_number=test_files,  dataset_type='TEST')
train_dataset = DroneThrmImg_Dataset(req_samples_number=train_files, dataset_type='TRAIN')


train_dataloader = DataLoader(train_dataset,shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True)

