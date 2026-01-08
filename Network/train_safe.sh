# Safe Training Script for Graph2Plan on ResPlan Dataset
# This script includes all NaN prevention measures

# Configuration
# - Gradient clipping: max_norm=5.0
# - Learning rate warmup: 3 epochs
# - NaN detection and skip for corrupted batches
# - Individual loss component NaN checks
# - Lower initial learning rate recommended

# RECOMMENDED: Start with lower learning rate
python train.py --batch_size 20 --epoch 150 --learning_rate 0.00005

# ALTERNATIVE: Original learning rate (riskier)
# python train.py --batch_size 20 --epoch 150 --learning_rate 0.0001

# Monitor the logs for:
# - "NaN/Inf detected" warnings (occasional is OK, frequent means deeper issue)
# - Loss should decrease gradually, not spike
# - Validation metrics should improve after epoch 3-5
