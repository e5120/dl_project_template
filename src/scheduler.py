from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
