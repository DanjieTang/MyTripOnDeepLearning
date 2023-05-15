import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

torch.manual_seed(1)

class MyFirstTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(784, 100)
        self.layer2=nn.Linear(100, 10)

    def forward(self, tensor):
        tensor=img_to_tensor(tensor).view(-1, 784)
        tensor=self.layer1(tensor)
        tensor=F.leaky_relu(tensor)
        tensor=self.layer2(tensor)
        return tensor

my_first_object=MyFirstTorch()

input=datasets.MNIST("data", train=True, download=True)
input=list(input)
train=input[:2000]
validation=input[2000:3000]
img_to_tensor=transforms.ToTensor()

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(my_first_object.parameters(), lr=0.005)

for (image, label) in train:
    actual=torch.zeros(1, 10)
    actual[0][label]=1
    output=my_first_object(image)

    loss=criterion(output, actual)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

mistake=0
for(image, label) in validation:
    actual=label
    output=my_first_object(image)
    predict=torch.argmax(output, dim=1)
    if(predict!=actual):
        mistake+=1


print("There is ", mistake, " mistakes in ", len(validation), " number of validation sets")