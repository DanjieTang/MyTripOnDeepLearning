import torch
import torch.nn as nn
import torch.optim as opt
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)

flag=False

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Conv2d(1, 3, 5, stride=2)
        self.layer2=nn.Conv2d(3, 6, 5, stride=2)
        self.layer3=nn.Conv2d(6, 12, 3)
        self.layer4=nn.ConvTranspose2d(12, 6, 3)
        self.layer5=nn.ConvTranspose2d(6, 3, 5, stride=2, output_padding=1)
        self.layer6=nn.ConvTranspose2d(3, 1, 5, stride=2, output_padding=1)

    def encoder(self, tensor):
        tensor=self.layer1(tensor)
        tensor=self.layer2(tensor)
        tensor=self.layer3(tensor)
        return tensor

    def decoder(self, tensor):
        tensor=self.layer4(tensor)
        tensor=self.layer5(tensor)
        tensor=self.layer6(tensor)
        return tensor

    def forward(self, tensor):
        tensor=self.encoder(tensor)
        if(flag):
            print(tensor)
            tensor[0]=torch.zeros(2, 2)
            tensor[1]=torch.zeros(2, 2)
            tensor[2]=torch.zeros(2, 2)
            print(tensor)
        tensor=self.decoder(tensor)

        return tensor

if __name__ == "__main__":
    hey=AutoEncoder()
    criterion=nn.MSELoss()
    optimzer=opt.Adam(hey.parameters(), lr=0.005)

    input=datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor())
    input=list(input)
    # train=input[:20000]
    #
    # for image, label in train:
    #     output=hey(image)
    #     loss=criterion(output, image)
    #     loss.backward()
    #
    #     optimzer.step()
    #     optimzer.zero_grad()

    hey.load_state_dict(torch.load("autoencoder_model.pth"))
    # torch.save(hey.state_dict(), "autoencoder_model.pth")

    #Display the image
    image=validation[0][0]
    image=image.view(28, -1)
    plt.imshow(image)

    #Pass the image through autoencoder
    image=image.view(1, 28, -1)
    output_image=hey(image)

    #Display the output image
    output_image=output_image.detach()
    output_image=output_image.view(28, -1)
    plt.imshow(output_image)
    plt.show()