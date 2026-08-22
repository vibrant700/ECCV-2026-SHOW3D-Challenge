"""
Joint-conditioned InterField model for the SHOW3D interaction-field challenge.

Inputs:
    images : (B, 3, 224, 224) RGB, ImageNet-normalized
    joints : (B, 2, 21, 3) hand joints in CAMERA frame, meters
             (hand order: 0 = left, 1 = right)

Output:
    fields : (B, 2, 21, 3) per-joint offset vectors in CAMERA frame, millimeters
             (vector from each hand joint to the nearest object-surface point)

The joint encoder uses a root-relative encoding (relative to the wrist joint) so
that the network sees hand-internal geometry plus the wrist's absolute position,
which keeps the numeric scale reasonable.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class JointInterFieldModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 512,
        joint_dim: int = 128,
        dropout: float = 0.5,
        num_hands: int = 2,
    ):
        super().__init__()

        # Image encoder (same as the official InterField baseline)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Joint encoder: per hand, relative(21*3=63) + wrist_root(3) = 66 -> joint_dim
        self.joint_encoder = nn.Sequential(
            nn.Linear(66, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, joint_dim),
        )

        self.num_hands = num_hands

        # Fused head: image features + both-hand joint features -> (num_hands, 21, 3)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + num_hands * joint_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_hands * 21 * 3),
        )

    def encode_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """joints: (B, 2, 21, 3) meters -> joint feature (B, 2 * joint_dim)."""
        root = joints[:, :, 0:1, :]          # (B, 2, 1, 3) wrist
        relative = joints - root             # (B, 2, 21, 3)
        joint_in = torch.cat(
            [relative.flatten(2), root.squeeze(2)], dim=-1
        )  # (B, 2, 66)
        joint_feat = self.joint_encoder(joint_in)  # (B, 2, joint_dim)
        return joint_feat.flatten(1)              # (B, 2 * joint_dim)

    def forward(self, images: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        img_feat = self.backbone(images)           # (B, 2048)
        joint_feat = self.encode_joints(joints)    # (B, 2 * joint_dim)
        fused = torch.cat([img_feat, joint_feat], dim=-1)
        out = self.head(fused)
        return out.view(B, self.num_hands, 21, 3)  # (B, 2, 21, 3) mm
