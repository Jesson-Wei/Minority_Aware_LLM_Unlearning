
# Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning

This is the codebase for our Project "Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning". The project involves fine-tuning models on three datasets and evaluating their vulnerability to membership inference attacks (MIA). Below, we outline the setup, dataset generation, key hyperparameters, and running instructions for fine-tuning and unlearning experiments.

## Environment Setup

We recommend setting up a `conda` environment for this project to ensure that all dependencies are installed correctly.

```bash
# Create and activate the environment
conda create -n llm_unlearning_minority python=3.10 
conda activate llm_unlearning_minority

# Navigate to the project directory
cd /path/to/project

# Install the project and dependencies in the file
pip install -e .
pip install scipy==1.10.1
pip install opacus==1.3.0

# Install PyTorch with CUDA support
pip uninstall torch -y
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install SpaCy and download the language model
pip install spacy
python -m spacy download en_core_web_sm
```

## Dataset Generation

To generate the datasets required for the experiments, you need to run the scripts available in the `dataset` folder. Each script corresponds to a specific dataset and will create the required dataset splits:

- `dataset_echr_year.py` generates the **ECHR_year** dataset.
- `dataset_enron_email.py` generates the **Enron_email** dataset.
- `dataset_enron_phone.py` generates the **Enron_phone** dataset.

These datasets will be used for fine-tuning and unlearning experiments.

## Fine-Tuning

To fine-tune the model without applying any unlearning methods, use the following command:

```bash
python fine_tune.py --config_path ../configs/fine-tune/finetune.yml
```

This command will fine-tune the pre-trained model on the selected dataset according to the configuration in `finetune.yml`.

## Unlearning

For running unlearning experiments, use the following command:

```bash
python fine_tune.py --config_path ../configs/unlearn/unlearn.yml
```

The unlearning configuration will apply the specified unlearning method (e.g., gradient ascent, scrub) to the dataset and model.

## MIA Attacks

To evaluate the trained or unlearned models using membership inference attacks (MIA), run the following:

```bash
python evaluation.py --config_path ../configs/mia/mia.yml
```

This will calculate the evaluation metrics to assess the model's vulnerability to MIA.

---
