BabyLM 2026: GPT-2 Baselines Using Interaction with a Teacher
=============================================================

This is the codebase that was used to run training for the Strict and Strict-Small baselines drawn from the 2025 Challenge's Interaction track. Please refer to the [Call for Papers](https://arxiv.org/pdf/2602.20092) for restrictions/rules around using a teacher model.

Installation
------------

To set up the environment, set up a virtual environment with uv (or any other environment manager of your choice) and install the requirements:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install requirements.txt
```

Data Processing and Training
----------------------------

The repository contains helper scripts to download the raw datasets for Strict and Strict-Small, process and clean them and then train tokenizers for each one of them. These are meant to be examples and you may change them, improve on them or approach each component completely differently. To replicate data processing, run the following:

```
bash experiment_scripts/prepare_data_and_tokenizers.sh
```

Training
--------
To replicate baseline training, run the following:

```
bash experiment_scripts/train.sh
```

Uploading to HF
---------------
We also have helper scripts for uploading trained models (and their intermediate checkpoints) to Huggingface. The models trained on this repo on the [BabyLM 2026 Baselines](https://huggingface.co/collections/BabyLM-community/babylm-2026-baselines) collection were uploaded with the following:

```
bash experiment_scripts/run_all_uploads.sh
```

As with all of the files in this repository, you can change this to fit your use-case.