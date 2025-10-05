# T5 Training on Azure for Students

## Your Azure Credits: $100 Free

## Cost-Optimized Training Plan

### Recommended Compute Options:
1. **Standard_NC6s_v3** (1x V100): ~$3.06/hour
2. **Standard_NC4as_T4_v3** (1x T4): ~$0.526/hour (BEST VALUE)
3. **Standard_NC6s_v2** (1x P100): ~$2.07/hour

### Training Time Estimates:
- **T4 GPU**: ~45-60 minutes for 3 epochs = **~$0.50**
- **V100 GPU**: ~20-30 minutes for 3 epochs = **~$1.50**
- **P100 GPU**: ~30-40 minutes for 3 epochs = **~$1.00**

## Step-by-Step Setup:

### 1. Access Azure Portal
- Go to https://portal.azure.com
- Sign in with your student account

### 2. Create Machine Learning Workspace
```bash
# In Azure Cloud Shell (free)
az group create --name rg-student-ml --location eastus
az ml workspace create --name student-ml-workspace --resource-group rg-student-ml
```

### 3. Upload Data
- Navigate to Azure ML Studio
- Go to Data > Datastores
- Upload your `train_20k_v2.jsonl` and `eval_20k_v2.jsonl`

### 4. Create Compute Instance
- **Recommended**: Standard_NC4as_T4_v3 (cheapest GPU option)
- **Location**: East US (usually cheapest)
- **Auto-shutdown**: Enable after 30 minutes idle

### 5. Training Command
```bash
python azure_train_t5.py \
  --batch-size 8 \
  --epochs 2 \
  --learning-rate 3e-4 \
  --model-name google-t5/t5-small
```

## Cost Optimization Tips:
1. **Use T5-small** first (60M params) - trains faster, costs less
2. **Enable auto-shutdown** on compute instances
3. **Use T4 GPU** instead of V100 for learning
4. **Start with 1-2 epochs** to test, then scale up

## Total Estimated Cost: $0.50 - $2.00

This leaves you $98+ credits for more experiments!

## Alternative: Free Options
1. **Google Colab Free**: 12-15 hour sessions with T4 GPU
2. **Kaggle Notebooks**: 30 hours/week free GPU
3. **Azure ML Serverless**: Pay only for compute time used 