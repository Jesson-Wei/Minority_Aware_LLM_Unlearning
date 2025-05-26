import os
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from datasets import load_from_disk
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.mia_attacks.mia_attack_factory import MIAAttackFactory
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.arguments.config_args import ConfigArgs
import transformers
from typing import Union

def parse_args():
    parser = transformers.HfArgumentParser((ConfigArgs,))
    return parser.parse_args_into_dataclasses()

def set_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and torch."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mia_scores(data, mia_attacker, seed):
    """Compute MIA attack scores for the given dataset."""
    set_seed(seed)
    return [mia_attacker.attack(data[i]) for i in tqdm(range(len(data)), desc="Computing criterion") if not np.isnan(mia_attacker.attack(data[i]))]

def calculate_auc(lm, dataset1, dataset2, mia_attacker, num_samples=100, seed=0):
    """Calculate AUC between two datasets using MIA attack."""
    set_seed(seed)
    sampled_dataset1 = random.sample([sample['text'] for sample in dataset1], num_samples)
    sampled_dataset2 = random.sample([sample['text'] for sample in dataset2], num_samples)
    scores1 = get_mia_scores(sampled_dataset1, mia_attacker, seed=seed)
    scores2 = get_mia_scores(sampled_dataset2, mia_attacker, seed=seed)
    inverted_scores1 = [-score for score in scores1]
    inverted_scores2 = [-score for score in scores2]
    labels = [1] * len(inverted_scores1) + [0] * len(inverted_scores2)
    scores = inverted_scores1 + inverted_scores2
    if not scores:
        raise ValueError("All scores are NaN after filtering!")
    auc = roc_auc_score(labels, scores)
    return auc, labels, scores

def calculate_tpr_at_fpr(labels, scores, target_fpr=0.05):
    """Calculate True Positive Rate (TPR) at a specific False Positive Rate (FPR)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_at_target_fpr = tpr[np.argmax(fpr >= target_fpr)]
    return tpr_at_target_fpr

def calculate_perplexity(model, dataset: Union[list, str], device: torch.device) -> float:
    """Calculate the perplexity using the model's internal perplexity function."""
    model._lm.eval()
    model._lm.to(device)
    ppl = model.perplexity(dataset, verbose=True)
    return ppl

def mia(config_args: ConfigArgs):
    """Perform Membership Inference Attack (MIA) on the dataset."""
    if config_args.exists():
        dataset_args = config_args.get_dataset_args()
        model_args = config_args.get_model_args()
        mia_attack_args = config_args.get_mia_attack_args()
        env_args = config_args.get_env_args()

    # Load the language model
    lm = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Create MIA attacker
    if mia_attack_args.attack_name == 'min_k':
        min_k_args = config_args.get_min_k_args()
        mia_attacker = MIAAttackFactory.from_args(mia_attack_args)(lm, min_k_args)
    else:
        mia_attacker = MIAAttackFactory.from_args(mia_attack_args)(lm)

    # Calculate AUC between poisoned dataset and test dataset
    auc_poisoned_vs_test, labels_poisoned_vs_test, scores_poisoned_vs_test = calculate_auc(lm, poisoned_dataset, test_dataset, mia_attacker, num_samples=200)
    print(f"AUC between poisoned_dataset and test_dataset: {auc_poisoned_vs_test:.3f}")

if __name__ == "__main__":
    # Use relative paths for datasets
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_directory = os.path.join(current_path, "source_dataset", "enron") # Adjust to the dataset to evaluate if needed

    # Load datasets and one can adjust to the dataset to evaluate
    original_train_dataset = load_from_disk(os.path.join(save_directory, "enron_phone_train"))
    poisoned_dataset = load_from_disk(os.path.join(save_directory, "enron_phone_perturb"))
    test_dataset = load_from_disk(os.path.join(save_directory, "enron_phone_test"))

    print(f"Loaded original_train_dataset with {len(original_train_dataset)} samples.")
    print(f"Loaded poisoned_dataset with {len(poisoned_dataset)} samples.")
    print(f"Loaded test_dataset with {len(test_dataset)} samples.")

    mia(*parse_args())
