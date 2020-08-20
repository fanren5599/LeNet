import matplotlib.pyplot as plt
from data import data_test_loader,data_train_loader
figure=plt.figure()
num_of_images=60

for imgs, targets in data_train_loader:
    break
print(imgs.shape)
for index in range(num_of_images):
    plt.subplot(6,10,index+1)
    plt.axis('off')
    img=imgs[index,...]
    print(img.shape)
    plt.imshow(img.numpy().squeeze(),cmap='gray_r')
plt.show()
