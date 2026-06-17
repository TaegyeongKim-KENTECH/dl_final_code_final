import torch
import torch.nn as nn
import open_clip

dict_pretrain = {
    'clipL14openai': ('ViT-L-14', 'openai'),
    'clipL14laion400m': ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B': ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp': ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool': ('ViT-L-14', 'laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp': ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B': ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B': ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b': ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B': ('ViT-B-16', 'laion2b_s34b_b88k'),
}


class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim):
        super(MultiScaleAttention, self).__init__()
        self.scale1 = nn.Linear(in_dim, in_dim)
        self.scale2 = nn.Linear(in_dim, in_dim)
        self.scale3 = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        return torch.sigmoid(s1 + s2 + s3)


class AFF(nn.Module):
    """Attentive Feature Fusion with multi-scale channel attention."""

    def __init__(self, in_dim):
        super(AFF, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(in_dim)
        self.ms_attention = MultiScaleAttention(in_dim)

    def forward(self, x1, x2):
        proj_query = self.query(x1)
        proj_key = self.key(x2)
        proj_value = self.value(x2)

        energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum('bij,bjk->bik', attention, proj_value)

        ms_attention = self.ms_attention(out)
        out = out * ms_attention
        return self.gamma * out + x1


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        proj_query = self.query(x)
        proj_key = self.key(x)
        proj_value = self.value(x)

        energy = torch.einsum('bik,bjk->bij', proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum('bij,bjk->bik', attention, proj_value)
        return self.gamma * out + x


class OpenClipLinear(nn.Module):
    """Baseline fake-image detector: frozen CLIP + AFF fusion + linear classifier."""

    def __init__(
        self,
        normalize=True,
        next_to_last=False,
        pretrained_model_path=None,
        num_classes=2,
        freeze_clip=True,
    ):
        super(OpenClipLinear, self).__init__()

        self.clip_model = open_clip.create_model(
            "ViT-L-14", pretrained=pretrained_model_path
        )
        if next_to_last:
            self.num_features = self.clip_model.visual.proj.shape[0]
            self.clip_model.visual.proj = None
        else:
            self.num_features = self.clip_model.visual.output_dim
        self.normalize = normalize

        self.aff = AFF(self.num_features)
        self.self_attention = SelfAttention(self.num_features)
        self.classifier = nn.Linear(self.num_features, 1)

        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _clip_encode(self, x):
        """Encode patches with frozen CLIP (no gradient)."""
        self.clip_model.eval()
        return self.clip_model.encode_image(x, normalize=self.normalize)

    def forward_features(self, x):
        return self._clip_encode(x)

    def forward(self, x):
        """Fuse local/global patch features and return logits [B, 1]."""
        batch_size = len(x)
        features_list = [self.forward_features(img) for img in x]

        fused_features = []
        for i in range(batch_size):
            local_feature1 = features_list[i][0, :].unsqueeze(0).unsqueeze(0)
            local_feature2 = features_list[i][1, :].unsqueeze(0).unsqueeze(0)
            global_feature = features_list[i][2, :].unsqueeze(0).unsqueeze(0)

            fused_local = self.aff(local_feature1, local_feature2)
            fused_feature = self.aff(fused_local, global_feature).squeeze(1)
            fused_features.append(fused_feature)

        fused_features = torch.stack(fused_features, dim=0)
        fused_features = self.self_attention(fused_features)
        return self.classifier(fused_features)

    def freeze_clip(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.aff.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_clip(self):
        for param in self.parameters():
            param.requires_grad = True
