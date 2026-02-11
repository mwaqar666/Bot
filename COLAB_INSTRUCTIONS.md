# AI Trading Bot - Google Colab Instructions

## 1. Setup Environment
Upload `crypto_bot_colab.zip` to the Colab Files area (left sidebar).

## 2. Run these cells in order

### Cell 1: Unzip and Install Dependencies
```python
!unzip -o crypto_bot_colab.zip
!pip install stable-baselines3[extra] pandas_ta shimmy gymnasium
```

### Cell 2: Verify GPU
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU! Enable GPU in Runtime > Change Runtime Type")
```

### Cell 3: Start TensorBoard (Run this BEFORE training)
```python
%load_ext tensorboard
%tensorboard --logdir ./tensorboard_logs/
```

### Cell 4: Run Training
```python
# Force python to find the modules
import sys
import os
sys.path.append(os.getcwd())

# Run the training script
!python ai_bot/train_agent.py
```

### Cell 5: Download Model
```python
from google.colab import files
!zip -r trained_model.zip ai_bot/models/
files.download('trained_model.zip')
```
