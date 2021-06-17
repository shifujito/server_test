import torch
from torch.backends import cudnn


print(torch.__version__)
print(torch.cuda.is_available())
# print(cudnn.getVersion())



print("Support CUDA ?: ", torch.cuda.is_available())
for _ in range(10000):
    x = torch.tensor([10.0])
    x = x.cuda()
    y = torch.randn(2, 3)
    y = y.cuda()
    z = x + y

    array = torch.zeros(4)
    array_gpu = array.cuda()
print(y)
print(x)  
print(z)

print(torch.cuda.device_count())
print("Support cudnn ?: ",cudnn.is_acceptable(x))