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
