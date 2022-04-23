# Mini-Project 1: Residual Network Design
## Members:
- Bhargav Makwana     : bm3125@nyu.edu
- Tanisha Madhusudhan : tm3805@nyu.edu
- Shubham Shandilya   : ss15590@nyu.edu

File ./DL_MiniProject_1.py contains our PyTorch description of our ResNet model architecture.
File ./checkpoint/ckpt.pt contains the trained weights of this architecture.

If required to run the model from starting:
execute: $ python DL_MiniProject.py

If required to resume the model from the saved checkpoint:
execute: $ python DL_MiniProject.py --resume

## Changes made to make self_eval.py compatible:
- Changed the filename and function name to project1_model in out model file.
- Changed the Normalization values according to our chosen one in the self_eval file.

# Mini-Project 2_3: Yoga Pose Detection
## Members:
- Bhargav Makwana     : bm3125@nyu.edu
- Tanisha Madhusudhan : tm3805@nyu.edu
- Shubham Shandilya   : ss15590@nyu.edu

## Architecture Used:
----------------------------------------------------------------\\
        Layer (type)               Output Shape         Param #\\
================================================================\\
            Conv2d-1         [-1, 64, 224, 224]           1,792\n
              ReLU-2         [-1, 64, 224, 224]               0\n
         MaxPool2d-3         [-1, 64, 112, 112]               0\n
           Dropout-4         [-1, 64, 112, 112]               0\n
            Conv2d-5        [-1, 128, 112, 112]          73,856\n
              ReLU-6        [-1, 128, 112, 112]               0\n
         MaxPool2d-7          [-1, 128, 56, 56]               0\n
           Dropout-8          [-1, 128, 56, 56]               0\n
            Conv2d-9          [-1, 256, 56, 56]         295,168\n
             ReLU-10          [-1, 256, 56, 56]               0\n
        MaxPool2d-11          [-1, 256, 28, 28]               0\n
          Dropout-12          [-1, 256, 28, 28]               0\n
           Linear-13                 [-1, 1024]     205,521,920\n
             ReLU-14                 [-1, 1024]               0\n
          Dropout-15                 [-1, 1024]               0\n
           Linear-16                    [-1, 5]           5,125\n
================================================================\n
Total params: 205,897,861\n
Trainable params: 205,897,861\n
Non-trainable params: 0\n
----------------------------------------------------------------\n
Input size (MB): 0.57\n
Forward/backward pass size (MB): 107.21\n
Params size (MB): 785.44\n
Estimated Total Size (MB): 893.22\n
----------------------------------------------------------------\n
### With Best Accuracy: 77.447%
