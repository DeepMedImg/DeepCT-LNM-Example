from torch import nn
from torch.nn import functional as F

class SoftmaxMseLoss(nn.Module):
    def __init__(self, size_average = False):
        super(SoftmaxMseLoss, self).__init__()
        self.size_average = size_average

    def forward(self, x, y):
        '''
        Take softmax on both sides and return MSE loss
        '''
        # assert x.size() == y.size()
        x_softmax = F.softmax(x, dim=1)
        y_softmax = F.softmax(y, dim=1)
        # batch_size, num_classes = x.size()[0], x.size()[1]
        return F.mse_loss(x_softmax, y_softmax)
        # return F.mse_loss(x_softmax, y_softmax, size_average=self.size_average) / (num_classes * batch_size)
