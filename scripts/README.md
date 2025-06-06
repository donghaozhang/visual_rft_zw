# Scripts Directory

This directory contains all the bash scripts for the Visual-RFT project, organized by functionality for training, inference, and setup operations.

## File Structure Overview
```
scripts/
├── README.md
├── setup/
│   └── setup.sh
├── training/
│   ├── core/
│   │   ├── qwen_2_5_dfew_zf_run_2.sh
│   │   └── qwen_2_5_dfew_zf_run.sh
│   ├── emotion_label/
│   │   ├── qwen_2_5_dfew_add_emotion_label_2.sh
│   │   └── qwen_2_5_dfew_add_emotion_label.sh
│   ├── reward_based/
│   │   ├── qwen_2_5_dfew_add_reward_for_emotion_label_2.sh
│   │   └── qwen_2_5_dfew_add_reward_for_emotion_label.sh
│   ├── coco/
│   │   ├── qwen_2_5_zf_run_2_coco.sh
│   │   ├── qwen_2_5_zf_run_coco.sh
│   │   ├── zf_run_2_29_04_2025_coco.sh
│   │   └── zf_run_29_04_2025_coco.sh
│   ├── legacy/
│   │   ├── dfew_zf_run_2_29_04_2025.sh
│   │   └── dfew_zf_run_29_04_2025.sh
│   └── interactive/
│       └── interaction_qwen_2_5_dfew_zf_run_2.sh
├── inference/
│   ├── run_2_inference_1_5_2025.sh
│   └── run_inference_1_5_2025.sh
└── distributed/
    ├── elastic_ddp_nccl_job.sh
    └── elastic_ddp_nccl.sh
```

## Table of Contents
- [Setup Scripts](#setup-scripts)
- [Training Scripts](#training-scripts)
- [Inference Scripts](#inference-scripts)
- [Distributed Training Scripts](#distributed-training-scripts)
- [Usage Guidelines](#usage-guidelines)

## Setup Scripts

### `setup/setup.sh`
**Purpose**: Environment setup and dependency installation
- Installs the Visual-RFT package in development mode
- Installs required dependencies: wandb, tensorboardx, qwen_vl_utils, flash-attn, vllm
- Fixes transformers version to specific commit

**Usage**:
```bash
bash scripts/setup/setup.sh
```

## Training Scripts

### Core Training Scripts
Located in `training/core/` - These scripts train the Qwen2.5-VL model using GRPO (Generalized Reward Policy Optimization) on the DFEW dataset.

#### `training/core/qwen_2_5_dfew_zf_run_2.sh`
**Purpose**: Main training script for Qwen2.5-VL-3B with DFEW dataset
- **Model**: Qwen2.5-VL-3B-Instruct
- **Dataset**: DFEW with emotion labels
- **Method**: GRPO training with reward clipping
- **Features**: Debug mode enabled, full offloading, cache disabled

**Usage**:
```bash
bash scripts/training/core/qwen_2_5_dfew_zf_run_2.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/core/qwen_2_5_dfew_zf_run.sh`
**Purpose**: Simplified version of the main training script
- Basic DFEW training without advanced features
- Similar configuration but smaller dataset

**Usage**:
```bash
bash scripts/training/core/qwen_2_5_dfew_zf_run.sh <nnodes> <nproc_per_node> <master_addr>
```

### Emotion Label Training Scripts
Located in `training/emotion_label/` - Scripts focused on emotion recognition training.

#### `training/emotion_label/qwen_2_5_dfew_add_emotion_label_2.sh`
**Purpose**: Training with enhanced emotion label processing
- **Dataset**: 2000 samples with emotion labels
- **Script**: Uses `grpo_qwen_2_5_dfew_reward_add_emotion_label.py`
- **Features**: Emotion-aware reward system

**Usage**:
```bash
bash scripts/training/emotion_label/qwen_2_5_dfew_add_emotion_label_2.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/emotion_label/qwen_2_5_dfew_add_emotion_label.sh`
**Purpose**: Basic emotion label training
- Standard emotion label integration
- Smaller dataset for testing

**Usage**:
```bash
bash scripts/training/emotion_label/qwen_2_5_dfew_add_emotion_label.sh <nnodes> <nproc_per_node> <master_addr>
```

### Reward-based Training Scripts
Located in `training/reward_based/` - Advanced reward systems for emotion recognition.

#### `training/reward_based/qwen_2_5_dfew_add_reward_for_emotion_label_2.sh`
**Purpose**: Advanced reward system for emotion recognition
- Incorporates reward mechanisms for emotion classification
- Enhanced training with emotion-specific rewards

**Usage**:
```bash
bash scripts/training/reward_based/qwen_2_5_dfew_add_reward_for_emotion_label_2.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/reward_based/qwen_2_5_dfew_add_reward_for_emotion_label.sh`
**Purpose**: Basic reward-based emotion training
- Simplified reward system for emotion labels

**Usage**:
```bash
bash scripts/training/reward_based/qwen_2_5_dfew_add_reward_for_emotion_label.sh <nnodes> <nproc_per_node> <master_addr>
```

### COCO Training Scripts
Located in `training/coco/` - Scripts for training on COCO dataset.

#### `training/coco/qwen_2_5_zf_run_2_coco.sh`
**Purpose**: Training on COCO dataset
- **Dataset**: COCO dataset for general vision tasks
- **Model**: Qwen2.5-VL with COCO-specific configuration

**Usage**:
```bash
bash scripts/training/coco/qwen_2_5_zf_run_2_coco.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `training/coco/qwen_2_5_zf_run_coco.sh`
**Purpose**: Basic COCO training script
- Simplified COCO training configuration

#### `training/coco/zf_run_2_29_04_2025_coco.sh` / `training/coco/zf_run_29_04_2025_coco.sh`
**Purpose**: Legacy COCO training scripts
- Historical versions of COCO training

### Legacy Training Scripts
Located in `training/legacy/` - Historical training scripts.

#### `training/legacy/dfew_zf_run_2_29_04_2025.sh`
**Purpose**: DFEW training script from April 29, 2025
- Legacy version with different configuration

#### `training/legacy/dfew_zf_run_29_04_2025.sh`
**Purpose**: Simplified legacy DFEW training

### Interactive Training
Located in `training/interactive/` - Enhanced monitoring and interactive features.

#### `training/interactive/interaction_qwen_2_5_dfew_zf_run_2.sh`
**Purpose**: Interactive training session for DFEW
- Enhanced logging and monitoring
- Interactive features for training observation

**Usage**:
```bash
bash scripts/training/interactive/interaction_qwen_2_5_dfew_zf_run_2.sh <nnodes> <nproc_per_node> <master_addr>
```

## Inference Scripts
Located in `inference/` - Model inference and evaluation scripts.

#### `inference/run_2_inference_1_5_2025.sh`
**Purpose**: Model inference on COCO dataset
- **Script**: Uses `Qwen2_VL_coco_infere_zw.py`
- **Configuration**: FP16 precision, batch size 1
- **Output**: Inference results with logging

**Usage**:
```bash
bash scripts/inference/run_2_inference_1_5_2025.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `inference/run_inference_1_5_2025.sh`
**Purpose**: Basic inference script
- Simplified inference configuration

**Usage**:
```bash
bash scripts/inference/run_inference_1_5_2025.sh <nnodes> <nproc_per_node> <master_addr>
```

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
bash scripts/training/reward_based/qwen_2_5_dfew_add_reward_for_emotion_label_2.sh 1 4 localhost

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