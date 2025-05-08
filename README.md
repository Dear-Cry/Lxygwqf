# 任务一：AlexNet/ResNet在Caltech-101上的微调实验与对比

本次实验的代码由以下Python文件构成，已上传至Github Repository（链接见文末）：
- main.py 加载数据，训练模型，绘制曲线
- mainTensorBoard.py 同main.py，但用TensorBoard代替matplotlib记录训练过程
- caltech101.py 加载Caltech101数据集的类
- plot.py 绘制Loss/Accuracy曲线的函数
- test.py 测试模型

训练模型只需在main.py中修改必要的参数并运行即可，测试模型需移步至test.py运行。
train.py会将训练得到的模型以二进制文件的形式保存为文件best_model.pth，
运行test.py时会从best_model.pth中加载模型并完成测试。

文件夹saved_model下保存了本报告中的三个模型，已上传至Google Drive（链接见文末），分别是对应
1.4, 1.5节中利用不同架构进行迁移学习的模型alexnet.pth, resnet18.pth, vgg16.pth.

Caltech101数据集在文件夹caltech-101下，已上传至Google Drive（链接见文末）.

实验报告已上传至elearning

Github Repository链接：

- https://github.com/Dear-Cry/Lxygwqf.git


Google Drive链接：

- https://drive.google.com/drive/folders/1ycJgqkhqbJm7vHAVdqFr2_IwjticjD25?usp=sharing
