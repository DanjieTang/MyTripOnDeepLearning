from torchvision.transforms.functional import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as opt
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN1=nn.Conv2d(1, 3, 3, stride=2)
        self.CNN2=nn.Conv2d(3, 5, 3, stride=2)
        self.CNN3=nn.Conv2d(5, 10, 3, stride=2)
        self.fc1=nn.Linear(40, 2)
        self.fc2=nn.Linear(40, 2)
        self.fc3=nn.Linear(2, 10*2*2)
        self.TCNN1=nn.ConvTranspose2d(10, 5, 3, stride=2, output_padding=1)
        self.TCNN2=nn.ConvTranspose2d(5, 3, 3, stride=2)
        self.TCNN3=nn.ConvTranspose2d(3, 1, 3, stride=2, output_padding=1)

    def forward(self, tensor):
        encoded=self.encode(tensor)
        mean, deviation=self.distribution(encoded)
        tensor=self.sampling(mean, deviation)
        decoded=self.decode(tensor)
        return encoded, mean, deviation, decoded

    def new_image(self, tensor):
        decoded=self.decode(tensor)
        return decoded

    def encode(self, tensor):
        tensor=self.CNN1(tensor)
        tensor=F.leaky_relu(tensor, 0.01)
        tensor=self.CNN2(tensor)
        tensor=F.leaky_relu(tensor, 0.01)
        tensor=self.CNN3(tensor)
        tensor=F.leaky_relu(tensor, 0.01)
        return tensor

    def distribution(self, tensor):
        tensor=tensor.view(-1, 1, 40)
        mean=self.fc1(tensor)
        deviation=self.fc2(tensor)
        return mean, deviation

    def sampling(self, mean, deviation):
        tensor=torch.randn(mean.size(0), mean.size(1), mean.size(2))
        tensor=mean+tensor*torch.exp(deviation/2.0)
        return tensor

    def decode(self, tensor):
        tensor=self.fc3(tensor)
        tensor=tensor.view(-1, 10, 2, 2)
        tensor=self.TCNN1(tensor)
        tensor=F.leaky_relu(tensor, 0.01)
        tensor=self.TCNN2(tensor)
        tensor=F.leaky_relu(tensor, 0.01)
        tensor=self.TCNN3(tensor)
        tensor=F.sigmoid(tensor)
        return tensor

criterion=nn.MSELoss()

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, 28, 28), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


if __name__ == "__main__":
    vae=VAE()
    optimizer=opt.Adam(vae.parameters(), lr=0.005)
    input=datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    input=list(input)
    train=input[:10000]
    validation=input[42000:43000]

    image_list=[0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(train)):
        if(i%8==0 and i!=0):
            image=torch.stack(image_list)
            encoded, mean, deviation, decoded=vae(image)
            loss=loss_function(decoded, image, mean, deviation)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        else:
            image_list[i%8]=train[i][0]



    image=validation[0][0]
    image=image.view(28, -1)
    # plt.imshow(image)
    # plt.show()
    image=image.view(1, 28, -1)

    image=vae(image)
    image=image[3]
    image=image.view(28, -1)
    image=image.detach().numpy()
    plt.imshow(image)
    plt.show()