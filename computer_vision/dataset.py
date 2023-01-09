import torch, torchvision, os, random, json
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

path = os.getcwd() + "/computer_vision/labels/"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.backends.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def count_all_jsons(): 
    total_json_number = 0
    for file in os.listdir(path):
        if file.endswith('.json'):
            total_json_number+=1
    return total_json_number

class DroneThrmImg_Dataset(Dataset):
    # create arrays, where random and non-repeating numbers of videos in the working directory will be stored 
    train_file_names, test_file_names = [], [] 
    random.seed(10)

    def __init__(self,req_samples_number, dataset_type):
        self.dataset_type = dataset_type
        self.total_json_number = count_all_jsons()
        
        if req_samples_number < self.total_json_number:
            self.req_samples_number = req_samples_number # number of samples required for training/test dataset and is less than total number of samples
        else:
            raise IndexError(f"required samples number ({req_samples_number}) is greater than\
 total number of available samples({self.total_json_number})")
        
        if self.dataset_type == 'TEST':
            self.random_videos = self.append_random_number(DroneThrmImg_Dataset.test_file_names,self.req_samples_number)

        elif self.dataset_type == 'TRAIN':
            self.random_videos = self.append_random_number(DroneThrmImg_Dataset.train_file_names,self.req_samples_number)

        else:
            raise NameError(f"'{dataset_type}' is not in the list of available dataset types. \nAvailable types are: 'TEST' and 'TRAIN.")
        

    def __len__(self):
        return self.req_samples_number # return required number of samples, since that's exactly length of the each dataset

    def __getitem__(self, idx):
        video_number = self.random_videos[idx] # take out random generated number of the video, according to its index
        label_name = "video_" + str(video_number) + ".json" # append this number to "video_" and ".json", like all other videos are named
        video_name = "video_" + str(video_number) + ".mp4"
        label_path = path+label_name # create complete path to the indexed video, i.e. "labels" of that video
        labels = open(file=label_path, mode="r")
        labels_json = json.load(labels)[str(video_number)] # adjust output of the json using [str(video_number)], to access labels
        video_tensor = torchvision.io.read_video(path+video_name)[0] # create video tensor for the output, which shape 
        
        # T,H,W,C = video_tensor.size()
        # 0,1,2,3

        gray_video_tensor = video_tensor.narrow(-1,0,1) # narrow last dimension(=channels) to 1
        #print(gray_video_tensor.size())
        gray_video_tensor = torch.permute(gray_video_tensor,(0,3,1,2))
        print(gray_video_tensor.size())
    
        labels = []
        for labeled_frame in labels_json: # each labeled frame in json has number of recogn. people and their coordinates, i.e. coordinates of label
            # create 16x16x3 grid, in each position of which is
            # the prob-ty of recognition and x,y-coordinates of the label to be stored
            grid = torch.zeros(16,16,3)             
            # since the number of people can vary 
            # make it sure that all labels in the grid are stored                 
            for recognized_person in labeled_frame: 
                row, column, x, y = labeled_frame[recognized_person]
                grid[int(row)][int(column)][0]=1 # probability that in this grid is person, make it to 100%
                grid[int(row)][int(column)][1]=x # x-coordinate of the label
                grid[int(row)][int(column)][2]=y # y-coordinate of the label
            labels.append(grid) # append that 16x16x3 gridded frame to the labels
        labels = torch.stack(labels)#.to(device=) # convert labels to the torch-type tensor
        print(labels.size())

        labels = labels.to(device).to(torch.float32)
        gray_video_tensor = gray_video_tensor.to(device).to(torch.float32)
        
        print(gray_video_tensor.dtype)
        print(labels.dtype)
        return gray_video_tensor, labels 

            
    def append_random_number(self, optional_list, samples_number):
        i = 0
        while i < samples_number:
            # substract 1 from total_json_number, because the first video is named using 0, not 1, 
            # so the number of the last video is 1 less then total quantity
            r = random.randint(0,self.total_json_number-1) 
            if not r in DroneThrmImg_Dataset.train_file_names \
                        and not r in DroneThrmImg_Dataset.test_file_names:
                optional_list.append(r)
                i+=1
        return optional_list


all_files = count_all_jsons()
train_files = int(0.8*all_files)
test_files = int(0.2*all_files)


test_dataset = DroneThrmImg_Dataset(req_samples_number=test_files,  dataset_type='TEST')
train_dataset = DroneThrmImg_Dataset(req_samples_number=train_files, dataset_type='TRAIN')


train_dataloader = DataLoader(train_dataset,shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True)


class myCNN:
    def __init__(self, input_channels):
        self.input_channels = input_channels

        self.model = torch.nn.Sequential(
            self.network()
        ).to(device)
        
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= 1e-3)
        
    def get_model(self):
        return self.model, self.loss_fn, self.optimizer

    def network(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(1,32, kernel_size=(7,11),stride=1), # = 250,310
            torch.nn.Conv2d(32,64, kernel_size=11,stride=2), # 120,150
            torch.nn.MaxPool2d(3), # 40,50
            torch.nn.Conv2d(64,128,kernel_size=2,stride=2), # 20,25
            torch.nn.Conv2d(128,128,kernel_size=(5,10),stride=1), # 16,16
            torch.nn.ReLU(), # 16,16
            torch.nn.Flatten(start_dim=0),
            torch.nn.Linear(32768,16*16*3),
            torch.nn.Sigmoid()
    )
    
    def get_train_data(self):
        train_dataloader = DataLoader(train_dataset,shuffle=True)
        return train_dataloader
    def get_test_data(self):
        test_dataloader = DataLoader(test_dataset,shuffle=True)
        return test_dataloader


my_cnn = myCNN(1)
model, loss, optimizer = my_cnn.get_model()


video, label = next(iter(my_cnn.get_train_data()))
video = video[0][0]
label = label[0][0]

print(torch.flatten(label).shape)
# summary(model.to(device), video.shape)
torch.flatten(label)
print(label.shape)
for i, batch in enumerate(iter(train_dataloader)):
    video, label = batch
    video = video[0]
    label = label[0]
    print(label.shape)

    print(label.get_device())
    print(label.get_device())
    for frame, labeled_frame in zip(video,label):
        # print(frame.shape)
        # print(labeled_frame.shape)
        # print(torch.flatten(labeled_frame).shape)
        pass

    conv1 = torch.nn.Conv2d(1,32, kernel_size=(7,11),stride=1).to(device)  # = 250,310
    conv2 = torch.nn.Conv2d(32,64, kernel_size=11,stride=2).to(device)  # 120,150
    pool  = torch.nn.MaxPool2d(3).to(device) # 40,50
    conv3 = torch.nn.Conv2d(64,128,kernel_size=2,stride=2).to(device) # 20,25
    conv4   = torch.nn.Conv2d(128,128,kernel_size=(5,10),stride=1).to(device) # 16,16
    relu    =  torch.nn.ReLU().to(device) # 16,16
    flatten = torch.nn.Flatten(start_dim=0).to(device)
    
    linear = torch.nn.Linear(32768,16*16*3).to(device)
    sigmoid = torch.nn.Sigmoid().to(device)
    for frame in video:
        print(frame.shape)

        x = conv1(frame)
        print(x.shape)
       
        x = conv2(x)
        print(x.shape)
        
        

        x = pool(x)
        print("pool: ",x.shape)

        x = conv3(x)
        print(x.shape)
        
        x = conv4(x)
        print(x.shape)
        x = relu(x)
        x = flatten(x)
  
        print("flatten: ", x.size())
       
       
        x = linear(x)
        x = sigmoid(x)
        print("linear: ",x.shape)

       
