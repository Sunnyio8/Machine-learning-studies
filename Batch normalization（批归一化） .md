# Batch normalization（批归一化）

Class: interview
Created: July 14, 2022 3:31 PM
Reviewed: No

## BN的概念和作用

在机器学习中，一般都会要求模型的输入分布是稳定的，如果不稳定或者训练集和测试集的分布不一致，就成为**协变量偏移**

在一个复杂的机器学习系统中，也会要求其中各个子部分的输入分布是稳定的，例如深度神经网络，在训练过程中，每一层的参数都与其之前的层有关系，当使用梯度下降更新参数时，当之前层的参数被更新，后一层输入数据的分布也会跟着变化，就为**内部协变量偏移**

网络越深，内部协变量偏移会给训练带来许多问题：

<aside>
💡 网络每一层训练更新参数时都需要不断适应输入数据的分布的变化，影响训练效率，并且使得学习过程变得不稳定

</aside>

<aside>
💡 为了尽量降低内部协变量偏移的影响，网络参数的更新需要更加谨慎，使得一般采用较小的学习率

</aside>

批归一化就是为了解决内部协变量偏移的问题而提出的，主要作用是使得每一层的参数发生了变化，输入输出数据的分布也不会产生较大变化，**使得训练过程更加稳定，避免了梯度爆炸和梯度消失，可以加快模型训练时的收敛速度**

批归一化的核心公式：

![Untitled](https://github.com/Sunnyio8/Machine-learning-studies/blob/main/images/Untitled.png)

## ****BN中均值、方差通过哪些维度计算得到？****

神经网络中传递的张量数据，其维度通常记为[N, H, W, C]，其中N是batch_size，H、W是行、列，C是通道数。那么上式中BN的输入集合就是下图中蓝色的部分。

![Untitled](Batch%20normalization%EF%BC%88%E6%89%B9%E5%BD%92%E4%B8%80%E5%8C%96%EF%BC%89%20704cf48e4f6a472d90abca45b157413e/Untitled%201.png)

均值的计算，就是在一个批次内，将每个通道中的数字单独加起来，再除以N×H×W。可训练参数的维度等于张量的通道数，RGB通道分别需要两个参数，因此维度等于3

## ****训练与推理时BN中的均值、方差分别是什么？****

**训练**时，均值、方差分别是**该批次**内数据相应维度的均值与方差；

**推理**时，均值、方差是**基于所有批次**的期望计算所得

BN层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。 在训练过程中，我们无法得知使用整个数据集来估计平均值和方差，所以只能根据每个小批次的平均值和方差不断训练模型。 而在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和方差。

 

### **注意：当batch size越小，BN的表现效果也越不好，因为计算过程中所得到的均值和方差不能代表全局**

## BN层的从零实现：

```python
def batch_norm(input_data, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (input_data - moving_mean)/torch.sqrt(moving_var - eps)
    else:
        assert len(input_data.shape) == (2, 4)
        if len(input_adta.shape) == 2:
        #全连接层
            mean = input_data.mean(dim=0)
            var = ((input_data - mean)**2).mean(dim=0)
        else:
            mean = input_data.mean(dim=0, keep_dim=True).mean(dim=2, keep_dim=True).mean(dim=3, keep_dim=True)
            var = ((input_data - mean)**2).mean(dim=0, keep_dim=True).mean(dim=2, keep_dim=True).mean(dim=3, keep_dim=True)
        X_hat = (input_data - mean)/torch.sqrt(var - eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma*X_hat + beta
    return Y, moving_mean, moving_var

class batchNorm(nn.Module):
    def __init__ (self, num_features, num_dims):
        super(batchNorm, self) __init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, x):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma, self.beta, self.moving_mean, self.moving_var, 1e-5, 0.9)
        return Y
```
