# OWL: Optimized Workforce Learning for General Multi-Agent Assistance for Real-World Task Automation

We present Workforce, a hierarchical multi-agent framework that decouples planning from execution through a modular
architecture with a domain-agnostic Planner, Coordinator, and specialized Workers. This enables cross-domain transfer by
allowing worker modification without full system retraining. Our OPTIMIZED WORKFORCE LEARNING (OWL) approach further
improves generalization through reinforcement learning. On the GAIA benchmark, Workforce achieves state-of-the-art
69.70% accuracy, outperforming commercial systems. Our OWL-trained 32B model reaches 52.73% accuracy, comparable to
GPT-4o. This work provides a foundation for general-purpose AI assistants with scalable domain adaptation.

This repository contains code for the OWL framework, including inference part (Workforce) and training part (OWL).

## Inference

To reproduce Workforce inference performance (69.70% - Claude-3.7 accuracy on GAIA benchmark and 60.61% - GPT-4o
accuracy on GAIA benchmark) shown in the paper, follow the steps below:

### Installation and Setup

1. Create a Python 3.11 Conda environment:

```bash
conda create -n owl python=3.11
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up envionment variables:

copy `.env.example` to `.env` and set the environment variables, and set the keys in `.env` file.

4. Run the inference:

- For reproducing results using GPT-4o, run:

```bash
python run_gaia_workforce.py
```

- For reproducing results using Claude-3.7, run:

```bash
python run_gaia_workforce_claude.py
```

## Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the Qwen2.5-32B-Instruct model.
For GPU resources, 8xH100 GPUs are recommended.

### Environment Setup

```shell
pip install openrlhf
pip install liger-kernel
cd train/LLaMA-Factory
```

```bash
python train/make_sft_data.py
python train/make_dpo_data.py --best_of_n 4 --level 1 --dataset gaia
```
- `--dataset` can be gaia, imm, wtq-xlsx, wtq-csv, math and hotpotqa

### Dataset Preparation

Download the dataset from [here](https://huggingface.co/datasets/anonymous21016/gaia_train_scored_planner).

### Model Training

1. For SFT stage, run:

```shell
llamafactory-cli train examples/train_full/all_qwen2_32B_planner_scored.yaml
```

1. For DPO stage, run:

```shell
deepspeed --include=localhost train_planner_dpo.py \
--save_path ./checkpoint/DPO_Qwen2.5-32B-Instruct_workforce_config2 \
--save_steps -1 \
--logging_steps 1 \
--eval_steps 10 \
--train_batch_size 128 \
--micro_train_batch_size 1 \
--pretrain Qwen/Qwen2.5-32B-Instruct \
--bf16 \
--max_epochs 2 \
--max_len 16384 \
--zero_stage 3 \
--beta 0.1 \
--learning_rate 5e-7 \
--dataset dataset/dpo_wtq-csv_workforce_1_config2_w_final_answer.json,dataset/dpo_wtq-xlsx_workforce_1_config2_w_final_answer.json \
--apply_chat_template \
--chosen_key chosen \
--rejected_key rejected \
--flash_attn \
--gradient_checkpointing \
--packing_samples \
--use_liger_kernel \
--adam_offload
```

Here we use `openrlhf` to train the model. The `--dataset` argument should point to the dataset you prepared in the
previous step.
The `--pretrain` argument should point to the pre-trained model you want to use.

### Evaluation

We use [vLLM](https://github.com/vllm-project/vllm) for LLM inference.

Here is an example command for running the inference:

```shell
vllm serve [YOUR_MODEL_PATH]  \
    --model-name [YOUR_MODEL_NAME] \
    --dtype=bfloat16 \
    --tensor-parallel-size=4 \
    --port 25001 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

python run_gaia_workforce_vllm_planner.py \
    --model_name [YOUR_MODEL_NAME] \
    --port 25001
```

