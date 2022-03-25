'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt


#####################################################################################
    # Residual Network Architecture    
#####################################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.dropout = nn.Dropout(0.25)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Our_ResNet():
    return ResNet(BasicBlock, [2, 1, 1, 1])

#####################################################################################
    # Instantiate and Build the Model
#####################################################################################
# Model
print('==> Building model..')
net = Our_ResNet()

#####################################################################################
    # Terminal Argument to parsed in order to decide whether to start from scracth 
    # or resume from the saved checkpoint
#####################################################################################
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
    
#####################################################################################
    # Append Dropout2d after each instance of BatchNorm2d 
    # without disturbing the architecture
#####################################################################################
'''
def append_dropout(net, rate=0.25):
    for name, module in net.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.BatchNorm2d):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(net, name, new)


append_dropout(net)
'''

#####################################################################################
    # Select and Configure the device
#####################################################################################
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
#####################################################################################
    # Initialize the variables
#####################################################################################  

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
test_loss_list = [] # save the test loss for each epoch to plot
test_acc_list = [] # to save the test accuracy for each epoch to plot

train_loss_list = [] # to save the train loss for each epoch
train_acc_list = [] # to save the train accuracy for each epoch to plot

#####################################################################################
    # Chain Various Data Augmentation techniques
#####################################################################################
# Data
print('==> Preparing data..')

transform_train = transforms.Compose([        
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
    
#####################################################################################
    # Download and Prepare the Data and split into train and test data-sets
#####################################################################################
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)


classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

#####################################################################################
    # Define functions to compute:
    # 1. Cross Entropy Loss
    # 2. Optimizer: SGD + Momentum
    # 3. CosineAnnealingLR Scheduler for Learning Rate            
#####################################################################################    
            
criterion = nn.CrossEntropyLoss()
#Adam Optimizer with L2 Regularization.
#optimizer = torch.optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-5)
#optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

'''
Set the learning rate of each parameter group using a cosine annealing schedule, 
where \eta_{max}is set to the initial lr and T_{cur} is the number of epochs since the last restart in SGDR:
'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#####################################################################################
    # Function to count the trainable parameters
#####################################################################################
    
def count_parameters(model):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
    # torch.numel() returns number of elements in a tensor
print(count_parameters(net))


#####################################################################################
    #Perform the Training and testing of the model
#####################################################################################
# Training
def train(epoch):
    global train_loss_list
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    train_loss_perEpoch = train_loss/(batch_idx+1)
    train_acc_perEpoch = 100.*correct/total
    train_loss_list.append(train_loss_perEpoch)
    train_acc_list.append(train_acc_perEpoch)
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss_perEpoch, train_acc_perEpoch, correct, total)  )


#####################################################################################
    
# Tesing
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    test_acc = 100.*correct/total
    test_loss_perEpoch = test_loss/(batch_idx+1)    
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss_perEpoch)
    if test_acc > best_acc:
        print('Saving..')
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss_perEpoch, test_acc, correct, total))
        print('#####################################################################################')
        
        #Save the state of the model, best accuracy yet and the current epoch
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pt')
        best_acc = test_acc

#####################################################################################
        # Call the Training and Testing functions 
        # based on the remaining number of epochs
#####################################################################################

for epoch in range(start_epoch, start_epoch+250):
    train(epoch)
    test(epoch)
    scheduler.step()

#####################################################################################

print('Best Accuracy of the model %.3f%%'% (best_acc))

plt.plot(train_loss_list)
plt.plot(test_loss_list)
plt.legend(["train", "val"])
plt.title("Loss")
plt.savefig("Last_Loss_4_4.jpg")
