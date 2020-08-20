import torch
import torch.nn as nn
from model import LeNet
from data import data_train_loader,data_test_loader

model_path="./model_save/model.pth"
save_info=torch.load(model_path)
model=LeNet()
criterion=nn.CrossEntropyLoss()
model.load_state_dict(save_info["model"])  #载入模型参数
model.eval() #切换模型到测试状态

test_loss=0
correct=0
total=0
with torch.no_grad():  #关闭计算图
    for batch_idx, (inputs,targets) in enumerate(data_test_loader):
        outputs=model(inputs)
        loss=criterion(outputs,targets)

        test_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()
        print(batch_idx,len(data_test_loader),'Loss:%.3f|Acc:%.3f%%(%d/%d)'%(test_loss/(batch_idx+1),100.*correct/total,correct,total))

