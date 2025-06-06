# Scripts Directory

This directory contains all the bash scripts for the Visual-RFT project, organized by functionality for training, inference, and setup operations.

## Table of Contents
- [Setup Scripts](#setup-scripts)
- [Training Scripts](#training-scripts)
- [Inference Scripts](#inference-scripts)
- [Distributed Training Scripts](#distributed-training-scripts)
- [Usage Guidelines](#usage-guidelines)

## Setup Scripts

### `setup.sh`
**Purpose**: Environment setup and dependency installation
- Installs the Visual-RFT package in development mode
- Installs required dependencies: wandb, tensorboardx, qwen_vl_utils, flash-attn, vllm
- Fixes transformers version to specific commit

**Usage**:
```bash
bash setup.sh
```

## Training Scripts

### Core Training Scripts
These scripts train the Qwen2.5-VL model using GRPO (Generalized Reward Policy Optimization) on the DFEW dataset.

#### `qwen_2_5_dfew_zf_run_2.sh`
**Purpose**: Main training script for Qwen2.5-VL-3B with DFEW dataset
- **Model**: Qwen2.5-VL-3B-Instruct
- **Dataset**: DFEW with emotion labels
- **Method**: GRPO training with reward clipping
- **Features**: Debug mode enabled, full offloading, cache disabled

**Usage**:
```bash
bash qwen_2_5_dfew_zf_run_2.sh <nnodes> <nproc_per_node> <master_addr>
```

#### `qwen_2_5_dfew_zf_run.sh`
**Purpose**: Simplified version of the main training script
- Basic DFEW training without advanced features
- Similar configuration but smaller dataset

### Emotion Label Training Scripts

#### `qwen_2_5_dfew_add_emotion_label_2.sh`
**Purpose**: Training with enhanced emotion label processing
- **Dataset**: 2000 samples with emotion labels
- **Script**: Uses `grpo_qwen_2_5_dfew_reward_add_emotion_label.py`
- **Features**: Emotion-aware reward system

#### `qwen_2_5_dfew_add_emotion_label.sh`
**Purpose**: Basic emotion label training
- Standard emotion label integration
- Smaller dataset for testing

### Reward-based Training Scripts

#### `qwen_2_5_dfew_add_reward_for_emotion_label_2.sh`
**Purpose**: Advanced reward system for emotion recognition
- Incorporates reward mechanisms for emotion classification
- Enhanced training with emotion-specific rewards

#### `qwen_2_5_dfew_add_reward_for_emotion_label.sh`
**Purpose**: Basic reward-based emotion training
- Simplified reward system for emotion labels

### COCO Training Scripts

#### `qwen_2_5_zf_run_2_coco.sh`
**Purpose**: Training on COCO dataset
- **Dataset**: COCO dataset for general vision tasks
- **Model**: Qwen2.5-VL with COCO-specific configuration

#### `qwen_2_5_zf_run_coco.sh`
**Purpose**: Basic COCO training script
- Simplified COCO training configuration

### Legacy Training Scripts

#### `dfew_zf_run_2_29_04_2025.sh`
**Purpose**: DFEW training script from April 29, 2025
- Legacy version with different configuration

#### `dfew_zf_run_29_04_2025.sh`
**Purpose**: Simplified legacy DFEW training

#### `zf_run_2_29_04_2025_coco.sh` / `zf_run_29_04_2025_coco.sh`
**Purpose**: Legacy COCO training scripts
- Historical versions of COCO training

### Interactive Training

#### `interaction_qwen_2_5_dfew_zf_run_2.sh`
**Purpose**: Interactive training session for DFEW
- Enhanced logging and monitoring
- Interactive features for training observation

## Inference Scripts

#### `run_2_inference_1_5_2025.sh`
**Purpose**: Model inference on COCO dataset
- **Script**: Uses `Qwen2_VL_coco_infere_zw.py`
- **Configuration**: FP16 precision, batch size 1
- **Output**: Inference results with logging

#### `run_inference_1_5_2025.sh`
**Purpose**: Basic inference script
- Simplified inference configuration

**Usage**:
```bash
bash run_2_inference_1_5_2025.sh <nnodes> <nproc_per_node> <master_addr>
```

## Distributed Training Scripts

#### `elastic_ddp_nccl_job.sh`
**Purpose**: Elastic distributed training job launcher
- Sets up distributed training environment
- Configures NCCL backend for multi-GPU training

#### `elastic_ddp_nccl.sh`
**Purpose**: Basic elastic distributed training
- **Backend**: NCCL with c10d
- **Framework**: PyTorch distributed training

**Usage**:
```bash
bash elastic_ddp_nccl.sh <nnodes> <nproc_per_node> <master_addr>
```

## Usage Guidelines

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
bash setup.sh

# Train model on DFEW dataset
bash qwen_2_5_dfew_zf_run_2.sh 1 4 localhost

# Run inference
bash run_2_inference_1_5_2025.sh 1 1 localhost

# Distributed training
bash elastic_ddp_nccl.sh 2 4 node1.cluster.com
```

### Notes
- All scripts include comprehensive logging to `logs/` directory
- Debug mode can be enabled for detailed training monitoring
- Scripts are optimized for the Gadi cluster environment
- Model paths and data paths may need adjustment for different environments 