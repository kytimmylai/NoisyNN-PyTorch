""" NoisyNN in PyTorch

'NoisyNN: Exploring the Influence of Information Entropy Change in Learning Systems'
- https://arxiv.org/pdf/2309.10625v2.pdf

Note that it's not an official implementation
"""
import torch
from timm.models.vision_transformer import VisionTransformer

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

class NoisyViT(VisionTransformer):
    """r
    Args:
        optimal: Determine the linear transform noise is produced by the quality matrix or the optimal quality matrix.
        res: Inference resolution. Ensure the aspect ratio = 1
    
    """
    def __init__(self, optimal: bool, res: int, **kwargs):
        self.stage3_res = res // 16
        if optimal:
            linear_transform_noise = optimal_quality_matrix(self.stage3_res)
        else:
            linear_transform_noise = quality_matrix(self.stage3_res)
        super().__init__(**kwargs)
        self.linear_transform_noise = torch.nn.Parameter(linear_transform_noise, requires_grad=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return super().forward_features(x)
        
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # Add noise when training/testing
        # See https://openreview.net/forum?id=Ce0dDt9tUT for more detail
        x = self.blocks[:-1](x)
        # Suppose the token dim = 1
        token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :].permute(0, 2, 1)
        B, C, L = x.shape
        x = x.reshape(B, C, self.stage3_res, self.stage3_res)
        x = self.linear_transform_noise@x + x
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = torch.cat([token, x], dim=1)
        x = self.blocks[-1](x)

        x = self.norm(x)
        return x

# We don't specify more args because the paper didn't reveal more details
def vit_t(optimal=True, res=224) -> NoisyViT:
    model = NoisyViT(
        optimal=optimal, 
        res=res, 
        patch_size=16, 
        embed_dim=192, 
        depth=12, 
        num_heads=3
    )
    return model

def vit_s(optimal=True, res=224) -> NoisyViT:
    model = NoisyViT(
        optimal=optimal, 
        res=res, 
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6
    )
    return model

def vit_b(optimal=True, res=224) -> NoisyViT:
    model = NoisyViT(
        optimal=optimal, 
        res=res, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12
    )
    return model

def vit_l(optimal=True, res=224) -> NoisyViT:
    model = NoisyViT(
        optimal=optimal, 
        res=res, 
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16
    )
    return model

# Easy test
if __name__ == '__main__':
    model = vit_l().cuda()
    inputs = torch.rand((2, 3, 224, 224)).cuda()
    output = model(inputs)
    print('Pass')