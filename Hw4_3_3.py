from matplotlib import pyplot as plt
import torch, torchvision
from visdom import Visdom
from torch import nn,optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

######################################
#    Part I : Write Data Loaders     #
######################################

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train_data = torchvision.datasets.MNIST('mnist_data',transform=T,download=True, train=True)
mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_data,batch_size=128)

mnist_test_data = torchvision.datasets.MNIST('mnist_data',transform=T,download=True, train=False)
mnist_test_dataloader = torch.utils.data.DataLoader(mnist_train_data,batch_size=1)



######################################
#    Part II : Write the Neural Net  #
######################################

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet,self).__init__()
        self.linear1 = nn.Linear(28*28,300)
        self.linear2 = nn.Linear(300,200)
        self.final_linear = nn.Linear(200,10)
        
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(1)
        
    def forward(self,images):
        x = images.view(-1,28*28)
        x = self.sig(self.linear1(x))
        x = self.sig(self.linear2(x))
        x = self.soft(self.final_linear(x))
        return x
    


# #####################################
#   Part III : Write Training Loop   #
# #####################################
# torch.Tensor.ndim = property(lambda self: len(self.shape)) 
model = Mnet()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params=params,lr=0.001)

n_epochs=3
n_iterations=0

# # vis=Visdom()
# # vis_window=vis.line(np.array([0]),np.array([0]))

for e in range(n_epochs):
    for i,(images,labels) in enumerate(mnist_train_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)
        
        
        model.zero_grad()
        loss = cec_loss(output,labels)
        # print(loss.data)
        # plt.plot(loss.data, n_iterations)
        # print(loss)
        loss.backward()
        
        optimizer.step()
        
        n_iterations+=1
        
        # vis.line(np.array([loss.item()]),np.array([n_iterations]),win=vis_window,update='append')
# plt.show()
# #####################################
#   Part IV : Write Testing Loop   #
# #####################################
test_loss = []
correct = 0


for i,(images,labels) in enumerate(mnist_test_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        output = model(images)
        predict = torch.argmax(output)
    
        if(predict==labels):
             test_loss.append(1)
        else:
             test_loss.append(0)

accuracy = sum(test_loss)/len(test_loss)
error = 1-accuracy
print("Error: ", error)