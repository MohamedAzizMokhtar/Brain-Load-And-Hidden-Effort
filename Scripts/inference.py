"""
Inference script for fMRI anomaly detection.
Loads a trained model and predicts if a new subject is anomalous.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# Model architecture (must match the trained model)
def _group_norm(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock3D(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        r = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.drop(x)
        x = self.conv2(F.silu(self.norm2(x)))
        return x + r


class StrongRes3DAE(nn.Module):
    def __init__(self, base_channels=24, dropout=0.0):
        super().__init__()
        c1, c2, c3, c4 = [base_channels * (2**i) for i in range(4)]

        self.stem = nn.Conv3d(1, c1, 3, padding=1)
        self.enc1 = ResBlock3D(c1, dropout)
        self.down1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ResBlock3D(c2, dropout)
        self.down2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.enc3 = ResBlock3D(c3, dropout)
        self.down3 = nn.Conv3d(c3, c4, 3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(ResBlock3D(c4, dropout), ResBlock3D(c4, dropout))

        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = ResBlock3D(c3, dropout)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ResBlock3D(c2, dropout)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ResBlock3D(c1, dropout)
        self.out = nn.Conv3d(c1, 1, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        xb = self.bottleneck(self.down3(x3))
        x = self.dec3(self.up3(xb) + x3)
        x = self.dec2(self.up2(x) + x2)
        x = self.dec1(self.up1(x) + x1)
        return torch.sigmoid(self.out(x))


class AnomalyDetector:
    """Load trained model and detect anomalies in new fMRI clips."""
    
    def __init__(self, checkpoint_path, device=None):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint (.pt file)
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config
        config = checkpoint.get('config', {'base_channels': 24, 'dropout': 0.0})
        self.threshold = checkpoint.get('threshold', 0.05)  # Default threshold
        
        # Build model
        self.model = StrongRes3DAE(
            base_channels=config.get('base_channels', 24),
            dropout=config.get('dropout', 0.0)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        print(f"? Model loaded on {self.device}")
        print(f"   Anomaly threshold: {self.threshold:.6f}")
    
    def score(self, clip):
        """
        Calculate anomaly score for a single fMRI clip.
        
        Args:
            clip: numpy array of shape (1, T, H, W) or (T, H, W)
                  Values should be in [0, 1] range (preprocessed)
        
        Returns:
            anomaly_score: float (higher = more anomalous)
            reconstruction: numpy array of same shape as input
            error_map: numpy array of same shape (pixel-wise error)
        """
        # Handle input shape
        if clip.ndim == 3:
            clip = clip[np.newaxis, ...]  # Add channel dimension
        if clip.ndim == 4 and clip.shape[0] != 1:
            clip = clip[np.newaxis, ...]  # Add batch dimension
        
        # Convert to tensor
        x = torch.from_numpy(clip.astype(np.float32)).to(self.device)
        
        # Add batch dimension if needed
        if x.ndim == 4:
            x = x.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            recon = self.model(x)
        
        # Calculate error
        error = F.mse_loss(recon, x, reduction='none')
        anomaly_score = error.mean().item()
        
        # Convert back to numpy
        recon_np = recon.squeeze().cpu().numpy()
        error_np = error.squeeze().cpu().numpy()
        
        return anomaly_score, recon_np, error_np
    
    def is_anomaly(self, score):
        """Return True if score exceeds threshold."""
        return score > self.threshold


# Example usage
if __name__ == "__main__":
    print("="*50)
    print("Inference Script for Brain-Load-And-Hidden-Effort")
    print("="*50)
    print("\nUsage example:")
    print("""
    from inference import AnomalyDetector
    
    # Load the trained model
    detector = AnomalyDetector('best_model_res3d_24.pt')
    
    # Load your preprocessed fMRI clip (shape: 1, 16, 64, 64)
    # clip = np.load('your_clip.npy')
    
    # Get anomaly score
    score, recon, error_map = detector.score(clip)
    
    # Check if anomalous
    if detector.is_anomaly(score):
        print(f"?? ANOMALY DETECTED! Score: {score:.4f}")
    else:
        print(f"? Normal brain pattern. Score: {score:.4f}")
    """)