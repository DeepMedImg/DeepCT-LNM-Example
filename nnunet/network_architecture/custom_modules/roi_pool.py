# Created by zhilin zheng
from copy import deepcopy
from nnunet.network_architecture.custom_modules.helperModules import Identity
from torch import nn
import numpy as np
import torch
from nnunet.utilities.to_torch import to_cuda

class MyRoIPooling3D(nn.Module):
    def __init__(self, scale, output_size):
        super(MyRoIPooling3D, self).__init__()
        self.scale = to_cuda(torch.from_numpy(scale).float())
        if len(self.scale.shape) == 1:
            self.scale = self.scale[:, None]

        self.output_size = output_size
        self.pool_op = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x, roi):
        '''
        params: x (batch, c, x ,y, z)
        params: roi (batch, 3, 2) --->[[minxidx, maxxidx], [minyidx, maxyidx], [minzidx, maxzidx]]
        '''
        # if type(roi) == np.ndarray:
        #     roi = torch.from_numpy(roi).float()
        roi = roi.data.clone()
        roi = torch.mul(roi, self.scale).long()
        # roi[:,:,1] += 1
        output = []
        for batch_idx in range(x.shape[0]):
            x1 = x[batch_idx:batch_idx+1, :, roi[batch_idx, 0, 0]: roi[batch_idx, 0, 1], roi[batch_idx, 1, 0]: roi[batch_idx, 1, 1], roi[batch_idx, 2, 0]: roi[batch_idx, 2, 1]]
            output.append(self.pool_op(x1))
        return torch.cat(output, 0)



if __name__ == "__main__":
    roi_pool = MyRoIPooling3D(np.array([0.25, 0.125, 0.5]), 1)
    x = to_cuda(torch.from_numpy(np.random.normal(size=(2,1, 16, 16, 16))))
    roi = to_cuda(torch.from_numpy(np.array([[[0,8],[4,6],[0,8]], [[2,8],[2,4],[2,8]]])))
    output = roi_pool(x, roi)
    print(output)
    test_output = x[:1,:, 0:2, 2:4, 0:4].cpu().numpy()
    test_output = np.mean(test_output, axis=(2,3,4), keepdims=True)
    print(test_output)