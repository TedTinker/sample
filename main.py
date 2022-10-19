#%%

import os
import cv2
import enlighten
from random import sample
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from utils import device
from arena import Arena
from model import Autoencoder 



arena = Arena(GUI = False)
arena.start()
autoencoder = Autoencoder() 

train_buffer = []
test_buffer = []

train_losses = [] 
test_losses  = []



def epoch(batch_size = 64, test = False): 
    buffer = test_buffer if test else train_buffer
    if(test): 
        batch = buffer
        autoencoder.train()
    else:     
        batch = sample(buffer, batch_size)
        autoencoder.eval()
    batch = np.array(batch)
    batch = torch.tensor(batch).to(device).to(torch.float)
    means, stdevs, decodings = autoencoder(batch.detach())
    
    # Make this make the means and stdevs closer to (1,0)!
    #loss = F.kl_div(
    #    torch.log(decodings), 
    #    torch.log(batch), 
    #    reduction="sum", log_target=True)
    
    loss = F.mse_loss(decodings, batch, reduction = "sum")
    
    autoencoder.optim.zero_grad()
    loss.backward()
    autoencoder.optim.step()
    
    losses = test_losses if test else train_losses
    losses.append(loss.item())
    
    
    
def example(epoch, quantity, save):
    images = np.array(test_buffer[:quantity])
    images = torch.tensor(images).to(device).to(torch.float)
    _, _, reconstruction = autoencoder(images)
        
    fig, axs = plt.subplots(2, 10, figsize = (quantity, 2.5))
    axs[0][0].title.set_text("Original Image")
    axs[1][0].title.set_text("Reconstruction")
    
    for c, ax in enumerate(axs[0]):
        ax.imshow(images[c].detach().cpu().squeeze(0))
        ax.axis("off")
        
    for c, ax in enumerate(axs[1]):    
        ax.imshow(reconstruction[c].detach().cpu().squeeze(0))
        ax.axis("off")
    
    fig.suptitle("Epoch {}".format(epoch))
    
    if(save): plt.savefig("saves/example_{}.png".format(str(epoch).zfill(6)))
    #plt.show()
    plt.close()
    
def losses(train_xs, test_xs, save):
    plt.title("Training and Test Losses")
    plt.xlabel = "Epoch"
    plt.ylabel = "Loss"
    plt.plot(train_xs, train_losses, label = "Training losses")
    plt.plot(test_xs,  test_losses,  label = "Test losses")
    plt.legend()
    if(save): plt.savefig("saves/losses.png")
    #plt.show()
    plt.close()
    
    
    
def train(
    epochs = 10000, batch_size = 128, 
    train_buffer_size = 1000, test_buffer_size = 256, 
    test_and_example = 10, example_quantity = 10):
    
    manager = enlighten.Manager()
    E = manager.counter(total = train_buffer_size, desc = "Training Buffer Images:", unit = "ticks", color = "red")
    for _ in range(train_buffer_size):
        train_buffer.append(arena.take_photo())
        E.update()
        
    manager = enlighten.Manager()
    E = manager.counter(total = test_buffer_size, desc = "Test Buffer Images:", unit = "ticks", color = "red")
    for _ in range(test_buffer_size):
        test_buffer.append(arena.take_photo())
        E.update()
    
    manager = enlighten.Manager()
    E = manager.counter(total = epochs, desc = "Epochs:", unit = "ticks", color = "blue")
    train_xs = []
    test_xs  = []
    for e in range(1, epochs+1):
        train_xs.append(e)
        epoch(batch_size = batch_size, test = False)
        if(e == 1 or e == epochs or e % test_and_example == 0):
            test_xs.append(e)
            epoch(batch_size = batch_size, test = True) 
            example(e, example_quantity, save = True)
            losses(train_xs, test_xs, save = True)
        E.update()
    
    files = [file for file in os.listdir("saves") if not file.split(".")[0] in ["losses", "video"]]
    files.sort()
            
    frame = cv2.imread("saves/" + files[0]); height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
    video = cv2.VideoWriter("saves/video.avi", fourcc, 1, (width, height))
    for file in files:
        video.write(cv2.imread("saves/" + file))
    cv2.destroyAllWindows()
    video.release()
    
    
    
import datetime

def duration():
    change_time = datetime.datetime.now() - start_time
    return(change_time)

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    train()
    print(duration())
# %%
