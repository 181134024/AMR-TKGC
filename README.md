# AMR-TKGC

## Overview

This project supports training on three datasets: **ProKG**, **WN18RR**, and **FB15k-237** using BERT as the pre-trained language model. You can choose from three different score functions: **ConvE**, **TransE**, or **DistMult**.

### Score Functions

The following score functions are supported:
- **ConvE**: Convolutional 2D Knowledge Graph Embeddings (default)
- **TransE**: Translating Embeddings for Modeling Multi-relational Data
- **DistMult**: Embedding Entities and Relations for Learning and Inference in Knowledge Bases

### Running Commands

Replace `$GPU_number` with your GPU ID (e.g., 0, 1, 2, ...).

---

## ProKG Dataset

### Training

**1. BERT-base with ConvE:**
```bash
python run.py -epoch 200 -name BERT_ProKG_ConvE -mi_drop -lr 1e-4 -batch 48 -test_batch 48 -num_workers 0 -k_w 10 -k_h 20 -embed_dim 200 -data ProKG -num_factors 2 -gpu $GPU_number -loss_weight --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**2. BERT-base with TransE:**
```bash
python run.py -epoch 200 -name BERT_ProKG_TransE -mi_drop -lr 1e-4 -batch 48 -test_batch 48 -num_workers 0 -k_w 10 -k_h 20 -embed_dim 200 -data ProKG -num_factors 2 -gpu $GPU_number -loss_weight -score_func TransE --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**3. BERT-base with DistMult:**
```bash
python run.py -epoch 200 -name BERT_ProKG_DistMult -mi_drop -lr 1e-4 -batch 48 -test_batch 48 -num_workers 0 -k_w 10 -k_h 20 -embed_dim 200 -data ProKG -num_factors 2 -gpu $GPU_number -loss_weight -score_func distmult --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

---

## WN18RR Dataset

### Training

**1. BERT-base with ConvE:**
```bash
python run.py -epoch 200 -name BERT_WN18RR_ConvE -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**2. BERT-base with TransE:**
```bash
python run.py -epoch 200 -name BERT_WN18RR_TransE -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight -score_func TransE --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**3. BERT-base with DistMult:**
```bash
python run.py -epoch 200 -name BERT_WN18RR_DistMult -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data WN18RR -num_factors 2 -gpu $GPU_number -loss_weight -score_func distmult --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

---

## FB15k-237 Dataset

### Training

**1. BERT-base with ConvE:**
```bash
python run.py -epoch 80 -name BERT_FB15k237_ConvE -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data FB15k-237 -num_factors 2 -gpu $GPU_number -loss_weight --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**2. BERT-base with TransE:**
```bash
python run.py -epoch 80 -name BERT_FB15k237_TransE -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data FB15k-237 -num_factors 2 -gpu $GPU_number -loss_weight -score_func TransE --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

**3. BERT-base with DistMult:**
```bash
python run.py -epoch 80 -name BERT_FB15k237_DistMult -mi_drop -lr 1e-4 -batch 64 -test_batch 64 -num_workers 4 -k_w 10 -k_h 20 -embed_dim 200 -data FB15k-237 -num_factors 2 -gpu $GPU_number -loss_weight -score_func distmult --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base
```

---

## Continue Training

To continue training from a checkpoint, add the following parameters to any training command:
- `-load_epoch $epoch`: The epoch number to load from
- `-load_type $checkpoint_type`: Checkpoint type ('combine', 'struc', or 'text')
- `-load_path $checkpoint_name`: The checkpoint name (model name + start time)

**Example (ProKG with TransE):**
```bash
python run.py -epoch 200 -name BERT_ProKG_TransE -mi_drop -lr 1e-4 -batch 48 -test_batch 48 -num_workers 0 -k_w 10 -k_h 20 -embed_dim 200 -data ProKG -num_factors 2 -gpu $GPU_number -loss_weight -score_func TransE --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name
```

## Testing

To test the model, add the `-test` parameter at the end of any training command:

**Example (ProKG with TransE):**
```bash
python run.py -epoch 200 -name BERT_ProKG_TransE -mi_drop -lr 1e-4 -batch 48 -test_batch 48 -num_workers 0 -k_w 10 -k_h 20 -embed_dim 200 -data ProKG -num_factors 2 -gpu $GPU_number -loss_weight -score_func TransE --enable_type_aware --type_mask_mode soft --type_mask_penalty 0.1 --type_prior_mode add --type_prior_alpha 0.04 --type_prior_tau 1.2 --dynamic_hops -confidence_threshold 0.6 -max_hops 3 -gcn_layer 3 -hop1_threshold 0.5 -hop2_threshold 0.4 -hop3_threshold 0.3 -pretrained_model bert_base -load_epoch $epoch -load_type $checkpoint_type -load_path $checkpoint_name -test
```

---

## Parameter Description

- `$GPU_number`: The GPU ID to use (e.g., 0, 1, 2, ...)
- `$epoch`: The specific epoch where training stopped
- `$checkpoint_type`: Model type, one of 'combine', 'struc', or 'text'
- `$checkpoint_name`: The model name you specified with `-name`, followed by the start time of first training

## Key Parameters

- `-data`: Dataset name ('ProKG', 'WN18RR', or 'FB15k-237')
- `-score_func`: Score function to use ('TransE' or 'distmult', default is ConvE)
- `-pretrained_model`: Pre-trained model ('bert_base' or 'bert_large')
- `--dynamic_hops`: Enable dynamic multi-hop reasoning
- `-confidence_threshold`: Confidence threshold for hop selection
- `-max_hops`: Maximum number of hops (default: 3)
- `-gcn_layer`: Number of GCN layers
- `-hop1_threshold`, `-hop2_threshold`, `-hop3_threshold`: Thresholds for different hop levels
- `--enable_type_aware`: Enable type-aware module
- `--type_prior_mode`: Type prior mode ('add' or 'mul')
- `--type_mask_mode`: Type mask mode ('soft' or 'hard')
- `--type_prior_alpha`: Type prior alpha value
- `--type_prior_tau`: Type prior tau value

---

