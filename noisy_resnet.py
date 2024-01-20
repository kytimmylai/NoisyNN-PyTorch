""" NoisyNN in PyTorch

'NoisyNN: Exploring the Influence of Information Entropy Change in Learning Systems'
- https://arxiv.org/pdf/2309.10625v2.pdf

Note that it's not an official implementation
"""
import torch
from timm.models.resnet import ResNet, BasicBlock, Bottleneck

def quality_matrix(k, alpha=0.3):
    """r
    Quality matrix Q. Described in the eq (17) so that eps = QX, where X is the input. 
    Alpha is 0.3, as mentioned in Appendix D.
    """
    identity = torch.diag(torch.ones(k))
    shift_identity = torch.zeros(k, k)
    for i in range(k):
        shift_identity[(i+1)%k, i] = 1
    opt = -alpha * identity + alpha * shift_identity
    return opt

def optimal_quality_matrix(k):
    """r
    Optimal Quality matrix Q. Described in the eq (19) so that eps = QX, where X is the input. 
    Suppose 1_(kxk) is torch.ones
    """
    return torch.diag(torch.ones(k)) * -k/(k+1) + torch.ones(k, k) / (k+1)

class NoisyResNet(ResNet):
    """r
    Args:
        optimal: Determine the linear transform noise is produced by the quality matrix or the optimal quality matrix.
        res: Inference resolution. Ensure the aspect ratio = 1
    
    """
    def __init__(self, optimal: bool, res: int, **kwargs):
        if optimal:
            linear_transform_noise = optimal_quality_matrix(res // 32)
        else:
            linear_transform_noise = quality_matrix(res // 32)
        super().__init__(**kwargs)
        self.linear_transform_noise = torch.nn.Parameter(linear_transform_noise, requires_grad=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return super().forward_features(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training:
            x = self.layer4[:-1](x)
            x = self.linear_transform_noise@x + x
            x = self.layer4[-1](x)
        else:
            x = self.layer4(x)

        return x


def res18(optimal=True, res=224) -> NoisyResNet:
    model = NoisyResNet(
        optimal=optimal, 
        res=res,  
        block=BasicBlock,
        layers=[2, 2, 2, 2]
    )
    return model

def res34(optimal=True, res=224) -> NoisyResNet:
    model = NoisyResNet(
        optimal=optimal, 
        res=res, 
        block=BasicBlock,
        layers=[3, 4, 6, 3]
    )
    return model

def res50(optimal=True, res=224) -> NoisyResNet:
    model = NoisyResNet(
        optimal=optimal, 
        res=res, 
        block=Bottleneck,
        layers=[3, 4, 6, 3]
    )
    return model

def res101(optimal=True, res=224) -> NoisyResNet:
    model = NoisyResNet(
        optimal=optimal, 
        res=res, 
        block=Bottleneck,
        layers=[3, 4, 23, 3]
    )
    return model

# Easy test
if __name__ == '__main__':
    model = res34().cuda()
    inputs = torch.rand((2, 3, 224, 224)).cuda()
    output = model(inputs)
    print('Pass')