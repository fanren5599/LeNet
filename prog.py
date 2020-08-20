import argparse

parser=argparse.ArgumentParser(description='PyTorch LeNet Training')
parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
parser.add_argument('--bath-size','-b',default=256,type=int,help='Batchsize')
args=parser.parse_args()
