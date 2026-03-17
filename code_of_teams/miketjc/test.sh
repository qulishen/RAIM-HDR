#!/bin/bash
# Test script for NTIRE 2026 RAIM MEF Phase 4 submission
# Verifies installation, dependencies, model loading, and inference

set -e

echo "================================"
echo "NTIRE 2026 RAIM MEF - Test Suite"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.10"
if [[ "$python_version" == 3.10* ]] || [[ "$python_version" == 3.11* ]] || [[ "$python_version" == 3.12* ]]; then
    echo -e "${GREEN}✓${NC} Python $python_version"
else
    echo -e "${YELLOW}⚠${NC} Python $python_version (recommended 3.10+)"
fi
echo ""

# Test 2: Check dependencies
echo "[2/5] Checking dependencies..."
python_check=$(python << 'EOF'
import sys
missing = []
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
except ImportError:
    missing.append("torch")

try:
    import torchvision
    print(f"  TorchVision: {torchvision.__version__}")
except ImportError:
    missing.append("torchvision")

try:
    import cv2
    print(f"  OpenCV: {cv2.__version__}")
except ImportError:
    missing.append("cv2")

try:
    import lpips
    print(f"  LPIPS: installed")
except ImportError:
    missing.append("lpips")

try:
    import einops
    print(f"  einops: installed")
except ImportError:
    missing.append("einops")

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    sys.exit(1)
EOF
)

if [ $? -eq 0 ]; then
    echo "$python_check"
    echo -e "${GREEN}✓${NC} All dependencies installed"
else
    echo -e "${RED}✗${NC} Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
fi
echo ""

# Test 3: Check GPU
echo "[3/5] Checking GPU..."
python << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
    print("\033[0;32m✓\033[0m GPU available")
else:
    print("\033[1;33m⚠\033[0m CPU only (inference will be very slow)")
EOF
echo ""

# Test 4: Check model checkpoint
echo "[4/5] Checking model checkpoint..."
checkpoint_path="checkpoints/raim_mef_phase2j/snapshot/net_final.pth"
if [ -f "$checkpoint_path" ]; then
    checkpoint_size=$(du -h "$checkpoint_path" | cut -f1)
    echo "  Found: $checkpoint_path ($checkpoint_size)"
    echo -e "${GREEN}✓${NC} Checkpoint exists"
else
    echo -e "${RED}✗${NC} Checkpoint not found: $checkpoint_path"
    exit 1
fi
echo ""

# Test 5: Test model loading and quick inference
echo "[5/5] Testing model loading and inference..."
python << 'EOF'
import sys
import os
import torch
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("  Loading model architecture...")
from models.archs.TimeDiffiT_ResNet_color_128_arch import TimeDiffiT_ResNet_color_128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    pass
args_obj = Args()
args_obj.in_channels = 15

model = TimeDiffiT_ResNet_color_128(dim=args_obj)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("  Loading checkpoint...")
checkpoint_path = "checkpoints/raim_mef_phase2j/snapshot/net_final.pth"
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

clean_state = OrderedDict()
for k, v in state_dict.items():
    key = k[7:] if k.startswith('module.') else k
    clean_state[key] = v

model.load_state_dict(clean_state, strict=False)
model = model.to(device)
model.eval()

print("  Model loaded successfully")
print(f"\033[0;32m✓\033[0m Model ready for inference")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗${NC} Model loading failed"
    exit 1
fi
echo ""

# All tests passed
echo "================================"
echo -e "${GREEN}✓ All tests passed!${NC}"
echo "================================"
echo ""
echo "You are ready to run:"
echo "  ./infer.sh          - Run inference on test data"
echo "  ./train.sh          - Run training (requires dataset)"
echo ""
