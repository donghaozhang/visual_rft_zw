# Scripts Directory

This directory contains all the bash scripts for the Visual-RFT project, organized by functionality for training, inference, and setup operations.

## File Structure Overview
```
scripts/
├── README.md
├── setup/
│   └── install_dependencies.sh
├── training/
│   ├── core/
│   │   ├── train_advanced.sh
│   │   └── train_basic.sh
│   ├── emotion_label/
│   │   ├── train_enhanced.sh
│   │   └── train_basic.sh
│   ├── reward_based/
│   │   ├── train_advanced_reward.sh
│   │   └── train_basic_reward.sh
│   ├── coco/
│   │   ├── train_advanced_coco.sh
│   │   ├── train_basic_coco.sh
│   │   ├── legacy_advanced_coco.sh
│   │   └── legacy_basic_coco.sh
│   ├── legacy/
│   │   ├── legacy_advanced_dfew.sh
│   │   └── legacy_basic_dfew.sh
│   └── interactive/
│       └── train_interactive.sh
├── inference/
│   ├── inference_advanced.sh
│   └── inference_basic.sh
└── distributed/
    ├── distributed_advanced.sh
    └── distributed_basic.sh
```

## Table of Contents
- [Setup Scripts](#setup-scripts)
- [Training Scripts](#training-scripts)
- [Inference Scripts](#inference-scripts)
- [Distributed Training Scripts](#distributed-training-scripts)
- [Usage Guidelines](#usage-guidelines)

## Setup Scripts

### `setup/install_dependencies.sh`
**Purpose**: Environment setup and dependency installation
- Installs the Visual-RFT package in development mode
- Installs required dependencies: wandb, tensorboardx, qwen_vl_utils, flash-attn, vllm
- Fixes transformers version to specific commit

**Usage**:
```bash
bash scripts/setup/install_dependencies.sh
```

## Training Scripts

### Core Training Scripts
Located in `training/core/` - These scripts train the Qwen2.5-VL model using GRPO (Generalized Reward Policy Optimization) on the DFEW dataset.

#### `training/core/train_advanced.sh`
**Purpose**: Main training script for Qwen2.5-VL-3B with DFEW dataset
- **Model**: Qwen2.5-VL-3B-Instruct
- **Dataset**: DFEW with emotion labels
- **Method**: GRPO training with reward clipping
- **Features**: Debug mode enabled, full offloading, cache disabled

**Usage**:
```bash
bash scripts/training/core/train_advanced.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/core/train_basic.sh`
**Purpose**: Simplified version of the main training script
- Basic DFEW training without advanced features
- Similar configuration but smaller dataset

**Usage**:
```bash
bash scripts/training/core/train_basic.sh <nnodes> <nproc_per_node> <master_addr>
```

### **Core Training Scripts Comparison**
| Feature | train_advanced.sh | train_basic.sh |
|---------|-------------------|----------------|
| **Complexity** | Advanced/Full-featured | Simplified/Basic |
| **Debug Mode** | Enabled by default | Standard logging |
| **Cache Usage** | Disabled (use_cache_false) | Default cache settings |
| **Offloading** | Full offloading enabled | Standard configuration |
| **Dataset** | Full DFEW with emotion labels | Basic DFEW dataset |
| **Reward System** | With clipping mechanism | Standard GRPO |
| **Recommended For** | Production training | Testing/Development |

### Emotion Label Training Scripts
Located in `training/emotion_label/` - Scripts focused on emotion recognition training.

#### `training/emotion_label/train_enhanced.sh`
**Purpose**: Training with enhanced emotion label processing
- **Dataset**: 2000 samples with emotion labels
- **Script**: Uses `grpo_qwen_2_5_dfew_reward_add_emotion_label.py`
- **Features**: Emotion-aware reward system

**Usage**:
```bash
bash scripts/training/emotion_label/train_enhanced.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/emotion_label/train_basic.sh`
**Purpose**: Basic emotion label training
- Standard emotion label integration
- Smaller dataset for testing

**Usage**:
```bash
bash scripts/training/emotion_label/train_basic.sh <nnodes> <nproc_per_node> <master_addr>
```

### **Emotion Label Training Scripts Comparison**
| Feature | train_enhanced.sh | train_basic.sh |
|---------|-------------------|----------------|
| **Dataset Size** | 2000 samples with emotion labels | Standard dataset size |
| **Script Used** | grpo_qwen_2_5_dfew_reward_add_emotion_label.py | Basic emotion label integration |
| **Reward System** | Enhanced emotion-aware rewards | Standard emotion labeling |
| **Training Intensity** | Intensive/Production-level | Basic/Testing |
| **Memory Usage** | Higher (larger dataset) | Lower (smaller dataset) |
| **Recommended For** | Full emotion recognition training | Initial testing/prototyping |

### Reward-based Training Scripts
Located in `training/reward_based/` - Advanced reward systems for emotion recognition.

#### `training/reward_based/train_advanced_reward.sh`
**Purpose**: Advanced reward system for emotion recognition
- Incorporates reward mechanisms for emotion classification
- Enhanced training with emotion-specific rewards

**Usage**:
```bash
bash scripts/training/reward_based/train_advanced_reward.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/reward_based/train_basic_reward.sh`
**Purpose**: Basic reward-based emotion training
- Simplified reward system for emotion labels

**Usage**:
```bash
bash scripts/training/reward_based/train_basic_reward.sh <nnodes> <nproc_per_node> <master_addr>
```

### **Reward-based Training Scripts Comparison**
| Feature | train_advanced_reward.sh | train_basic_reward.sh |
|---------|--------------------------------------------------|------------------------------------------------|
| **Reward Complexity** | Advanced reward mechanisms | Basic reward system |
| **Emotion Classification** | Enhanced reward for emotion classes | Simplified reward for emotions |
| **Training Approach** | Multi-faceted reward system | Single reward mechanism |
| **Performance** | Higher accuracy potential | Faster training |
| **Computational Cost** | Higher (complex rewards) | Lower (simple rewards) |
| **Recommended For** | Research/Production | Quick experiments |

### COCO Training Scripts
Located in `training/coco/` - Scripts for training on COCO dataset.

#### `training/coco/train_advanced_coco.sh`
**Purpose**: Training on COCO dataset
- **Dataset**: COCO dataset for general vision tasks
- **Model**: Qwen2.5-VL with COCO-specific configuration

**Usage**:
```bash
bash scripts/training/coco/train_advanced_coco.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/coco/train_basic_coco.sh`
**Purpose**: Basic COCO training script
- Simplified COCO training configuration

#### `training/coco/legacy_advanced_coco.sh`
**Purpose**: Legacy COCO training script (April 29, 2025)
- Historical version with advanced configuration

#### `training/coco/legacy_basic_coco.sh`
**Purpose**: Legacy COCO training script (April 29, 2025)
- Historical version with basic configuration

### **COCO Training Scripts Comparison**
| Feature | train_advanced_coco.sh | train_basic_coco.sh | legacy_advanced_coco.sh | legacy_basic_coco.sh |
|---------|----------------------------|--------------------------|------------------------------|----------------------------|
| **Version** | Current/Latest | Current/Basic | Legacy (April 2025) | Legacy (April 2025) |
| **Complexity** | Advanced features | Simplified | Advanced legacy | Basic legacy |
| **Model** | Qwen2.5-VL optimized | Qwen2.5-VL basic | Historical model | Historical model |
| **Configuration** | Full COCO setup | Basic COCO | Legacy advanced | Legacy basic |
| **Recommended For** | Production COCO training | Testing COCO | Historical reference | Historical testing |

### Legacy Training Scripts
Located in `training/legacy/` - Historical training scripts.

#### `training/legacy/legacy_advanced_dfew.sh`
**Purpose**: DFEW training script from April 29, 2025
- Legacy version with different configuration

#### `training/legacy/legacy_basic_dfew.sh`
**Purpose**: Simplified legacy DFEW training

### **Legacy Training Scripts Comparison**
| Feature | legacy_advanced_dfew.sh | legacy_basic_dfew.sh |
|---------|------------------------------|----------------------------|
| **Complexity** | Advanced legacy configuration | Basic legacy configuration |
| **Training Features** | Full feature set (April 2025) | Simplified features |
| **Historical Value** | Complete reference implementation | Minimal reference |
| **Use Case** | Understanding advanced legacy setup | Quick legacy testing |
| **Documentation** | Comprehensive legacy example | Basic legacy example |

### Interactive Training
Located in `training/interactive/` - Enhanced monitoring and interactive features.

#### `training/interactive/train_interactive.sh`
**Purpose**: Interactive training session for DFEW
- Enhanced logging and monitoring
- Interactive features for training observation

**Usage**:
```bash
bash scripts/training/interactive/train_interactive.sh <nnodes> <nproc_per_node> <master_addr>
```

## Inference Scripts
Located in `inference/` - Model inference and evaluation scripts.

#### `inference/inference_advanced.sh`
**Purpose**: Model inference on COCO dataset
- **Script**: Uses `Qwen2_VL_coco_infere_zw.py`
- **Configuration**: FP16 precision, batch size 1
- **Output**: Inference results with logging

**Usage**:
```bash
bash scripts/inference/inference_advanced.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `inference/inference_basic.sh`
**Purpose**: Basic inference script
- Simplified inference configuration

**Usage**:
```bash
bash scripts/inference/inference_basic.sh <nnodes> <nproc_per_node> <master_addr>
```

### **Inference Scripts Comparison**
| Feature | inference_advanced.sh | inference_basic.sh |
|---------|------------------------------|----------------------------|
| **Precision** | FP16 precision | Default precision |
| **Batch Size** | Optimized (batch size 1) | Basic batch size |
| **Logging** | Enhanced inference logging | Standard logging |
| **Performance** | Optimized for production | Basic inference |
| **Output Detail** | Comprehensive results | Standard results |
| **Recommended For** | Production inference | Quick testing |

## Distributed Training Scripts
Located in `distributed/` - Multi-GPU and multi-node training setup.

#### `distributed/elastic_ddp_nccl_job.sh`
**Purpose**: Elastic distributed training job launcher
- Sets up distributed training environment
- Configures NCCL backend for multi-GPU training

**Usage**:
```bash
bash scripts/distributed/elastic_ddp_nccl_job.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `distributed/elastic_ddp_nccl.sh`
**Purpose**: Basic elastic distributed training
- **Backend**: NCCL with c10d
- **Framework**: PyTorch distributed training

**Usage**:
```bash
bash scripts/distributed/elastic_ddp_nccl.sh <nnodes> <nproc_per_node> <master_addr>
```

### **Distributed Training Scripts Comparison**
| Feature | elastic_ddp_nccl_job.sh | elastic_ddp_nccl.sh |
|---------|-------------------------|---------------------|
| **Functionality** | Job launcher/coordinator | Direct training script |
| **Setup Complexity** | Advanced job management | Basic distributed setup |
| **Environment Config** | Comprehensive env setup | Minimal setup |
| **Multi-GPU Support** | Advanced GPU coordination | Basic GPU support |
| **Job Management** | Full job lifecycle | Simple execution |
| **Recommended For** | Production clusters | Development/Testing |

## Usage Guidelines

### Quick Navigation
Use these commands to navigate to specific script categories:
```bash
# Setup scripts
cd scripts/setup/

# Core training scripts
cd scripts/training/core/

# Emotion label training
cd scripts/training/emotion_label/

# Reward-based training
cd scripts/training/reward_based/

# COCO training
cd scripts/training/coco/

# Legacy training scripts
cd scripts/training/legacy/

# Interactive training
cd scripts/training/interactive/

# Inference scripts
cd scripts/inference/

# Distributed training scripts
cd scripts/distributed/
```

### Common Parameters
Most training and inference scripts accept these parameters:
- `<nnodes>`: Number of nodes (typically 1 for single-node training)
- `<nproc_per_node>`: Number of processes per node (usually equals number of GPUs)
- `<master_addr>`: Master node address for distributed training

### Environment Requirements
All scripts assume:
- **Cluster**: Gadi cluster environment
- **CUDA**: Version 12.5.1
- **GCC**: Version 12.2.0
- **Conda Environment**: Visual-RFT
- **Python Path**: `/scratch/kf09/zw4360/miniconda3/`

### Directory Structure Expected
```
./share_data/          # Datasets
./share_models/        # Model checkpoints
./logs/               # Training logs
./debug_logs/         # Debug output
```

### Key Features
- **DeepSpeed Integration**: Zero3 offloading for memory efficiency
- **Mixed Precision**: BF16/FP16 support
- **Gradient Checkpointing**: Memory optimization
- **WandB Integration**: Experiment tracking (offline mode)
- **Flexible Batch Sizes**: Configurable per-device batch sizes

### Example Usage
```bash
# Setup environment
bash scripts/setup/setup.sh

# Train model on DFEW dataset (core training)
bash scripts/training/core/qwen_2_5_dfew_zf_run_2.sh 1 4 localhost

# Train with emotion labels
bash scripts/training/emotion_label/qwen_2_5_dfew_add_emotion_label_2.sh 1 4 localhost

# Train with reward system
bash scripts/training/reward_based/train_advanced_reward.sh 1 4 localhost

# Train on COCO dataset
bash scripts/training/coco/qwen_2_5_zf_run_2_coco.sh 1 4 localhost

# Run inference
bash scripts/inference/run_2_inference_1_5_2025.sh 1 1 localhost

# Distributed training
bash scripts/distributed/elastic_ddp_nccl.sh 2 4 node1.cluster.com
```

### Notes
- All scripts include comprehensive logging to `logs/` directory
- Debug mode can be enabled for detailed training monitoring
- Scripts are optimized for the Gadi cluster environment
- Model paths and data paths may need adjustment for different environments
- Use the organized folder structure to easily find scripts by functionality 