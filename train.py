import torch
import torch.nn as nn
from model import LeNet
import torch.optim as optim
from data import data_train_loader,data_test_loader
from prog import args
model=LeNet()
model.train()  #切换模型到训练状态
#lr=0.01
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=5e-4)
epoch=5
for epoch_id in range(epoch):
    train_loss=0
    correct=0
    total=0
    for batch_idx, (inputs,targets) in enumerate(data_train_loader):
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _,predicted=outputs.max(1)
        total+=targets.size(0)
        correct+=predicted.eq(targets).sum().item()

        print(epoch_id, batch_idx,len(data_train_loader),'Loss:%.3f|Acc:%.3f%%(%d/%d)'%(train_loss/(batch_idx+1),100.*correct/total,correct,total))

save_info={
    "iter_num":epoch,                #迭代步数
    "optimizer":optimizer.state_dict(),  #优化器的状态字典
    "model":model.state_dict(),  #模型的状态字典
}
save_path="./model_save/model.pth"
#保存信息
torch.save(save_info,save_path)
