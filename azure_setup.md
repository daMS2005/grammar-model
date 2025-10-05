# T5 Fine-tuning on Azure Machine Learning

## Prerequisites
1. Azure account with subscription
2. Azure ML workspace created

## Setup Steps

### 1. Create Azure ML Workspace
```bash
# Install Azure CLI and ML extension
az extension add -n ml

# Create resource group
az group create --name rg-ml-punct --location eastus

# Create ML workspace
az ml workspace create --name ml-punct-workspace --resource-group rg-ml-punct
```

### 2. Upload Data to Azure
- Upload your `data/train_20k_v2.jsonl` and `data/eval_20k_v2.jsonl` to Azure ML datastore

### 3. Compute Instance
- Create a compute instance with GPU (Standard_NC6s_v3 or Standard_NC12s_v3)
- Or use serverless compute for training jobs

### 4. Training Script Modifications
The current `fine_tune_t5_punct.py` should work with minimal changes on Azure ML.

## Estimated Costs
- **Standard_NC6s_v3** (1 x V100): ~$3-4/hour
- **Standard_NC12s_v3** (2 x V100): ~$6-8/hour
- Training time: 30-60 minutes for 1 epoch

## Alternative: Google Colab Pro
If you want a quicker setup:
- $10/month for Colab Pro
- Access to V100/A100 GPUs
- Can run for several hours 