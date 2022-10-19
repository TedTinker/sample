#%%
import torch
from torch import nn 
import torch.optim as optim
import torchgan.layers as gnn
from torchinfo import summary as torch_summary

from utils import device, ConstrainedConv2d, init_weights, image_size



class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 3, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            
            nn.Dropout(.2),
                        
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            
            nn.Dropout(.2),
                        
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)),
            
            nn.Dropout(.2),
            
            gnn.SelfAttention2d(
                input_dims = 32, 
                output_dims = 32))
        
        example = torch.zeros([1, 3, image_size, image_size])
        example = self.image_in(example).flatten(1)
        
        self.mean = nn.Sequential(
            nn.Linear(example.shape[1], 1024),
            nn.LeakyReLU())
        
        self.stdev = nn.Sequential(
            nn.Linear(example.shape[1], 1024),
            nn.ReLU())

        self.decode = nn.Sequential(
            nn.Linear(1024, example.shape[1]),
            nn.LeakyReLU())

        self.image_out = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear", align_corners=True),
            
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear", align_corners=True),
            
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear", align_corners=True),
            
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 3,
                kernel_size = (1,1)))

        self.image_in.apply(init_weights)
        self.mean.apply(init_weights)
        self.stdev.apply(init_weights)
        self.decode.apply(init_weights)
        self.image_out.apply(init_weights)
        self.to(device)
        
        self.optim = optim.Adam(params=self.parameters(), lr=.001) 

    def forward(self, image):
        image = torch.permute(image, (0, -1, 1, 2)) * 2 - 1
        image = self.image_in(image).flatten(1)
        means = self.mean(image)
        stdevs = self.stdev(image)
        encoding = torch.normal(means, stdevs)
        image = self.decode(encoding).reshape((image.shape[0], 32, image_size//8, image_size//8))
        #image = self.decode(means).reshape((image.shape[0], 32, 16, 16))
        image = self.image_out(image)
        image = torch.clamp(image, -1, 1)
        image = (torch.permute(image, (0, 2, 3, 1)) + 1) / 2
        return(means, stdevs, image)



if __name__ == "__main__":

    autoencoder = Autoencoder()

    print("\n\n")
    print(autoencoder)
    print()
    print(torch_summary(autoencoder, (1, image_size, image_size, 3)))
    
print("models.py loaded.")
# %%