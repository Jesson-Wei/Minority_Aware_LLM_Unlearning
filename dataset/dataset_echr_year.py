import os
import re
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset, Dataset
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import random

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Regular expression to match various date formats, including numerical months
date_pattern = re.compile(r'''
    (?:\d{1,2}(?:st|nd|rd|th)?\s)?          # Optional day with suffix (e.g., "23rd", "1st")
    (?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|
           Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|
           Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?|0?[1-9]|1[0-2])     # Month names or numerical months
    (?:\s\d{1,2}(?:st|nd|rd|th)?)?         # Optional day without leading zero
    \s?                                    # Optional space
    (?P<year>\d{4})                        # Four-digit year
    ''', re.IGNORECASE | re.VERBOSE)

def extract_year_month(date_str):
    """Extract year and month from a date string using regex."""
    months_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
        'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    match = date_pattern.search(date_str)
    if match:
        year = match.group('year')
        month = match.group('month')
        try:
            year_int = int(year)
            if not (1900 <= year_int <= 2024):
                return None, None
        except ValueError:
            return None, None
        
        if month:
            month_cap = month.capitalize()
            if month_cap[:3] in months_map:
                return year, months_map[month_cap[:3]]
            if month.isdigit():
                month_int = int(month)
                if 1 <= month_int <= 12:
                    return year, f"{month_int:02d}"
        return year, '00'
    return None, None

def main():
    # Load the ECHR dataset
    print("Loading ECHR dataset...")
    extracted_dataset = load_dataset("echr")

    # Randomly sample 50,000 training samples, 50,000 evaluation samples, and r minority samples
    total_samples = len(extracted_dataset['train'])
    random_indices = np.random.permutation(total_samples)

    r = 100  # Set the minority set size
    total_num = 10000 + r
    train_indices = random_indices[:total_num]
    eval_indices = np.random.permutation(len(extracted_dataset['validation']))[:10000]

    train_dataset = extracted_dataset['train'].select(train_indices)
    eval_dataset = extracted_dataset['validation'].select(eval_indices)

    # Create a remain set
    all_train_indices = set(range(len(extracted_dataset['train'])))
    train_indices_set = set(train_indices)
    remain_indices = list(all_train_indices - train_indices_set)
    remain_set = extracted_dataset['train'].select(remain_indices)

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"Remain set size: {len(remain_set)}")

    # Initialize Flair NER model
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast").to('cuda')

    # Initialize counters
    year_counter = Counter()
    year_to_indices = {}

    # Process each sample in the train dataset
    for idx, case in enumerate(tqdm(train_dataset, desc="Processing training samples")):
        sentence = Sentence(case['text'])
        
        # Use Flair NER to extract entities
        tagger.predict(sentence)
        
        # Count years from extracted entities
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'DATE':
                year_match = re.search(r'(19|20)\d{2}', entity.text)
                if year_match:
                    year = year_match.group(0)
                    year_counter[year] += 1
                    if year not in year_to_indices:
                        year_to_indices[year] = []
                    year_to_indices[year].append(idx)

    # Sort years by least frequent and select minority samples
    least_frequent_years = [year for year, count in year_counter.most_common()][::-1]
    minority_size = r
    minority_indices = []

    # Add samples from the least frequent years
    samples_per_year = 1
    for year in least_frequent_years:
        available_indices = year_to_indices[year]
        for idx in available_indices[:samples_per_year]:
            minority_indices.append(idx)
        if len(minority_indices) >= minority_size:
            break

    # Ensure we have exactly r minority samples
    if len(minority_indices) > minority_size:
        minority_indices = minority_indices[:minority_size]

    print(f"Selected {len(minority_indices)} samples for the minority set.")

    # Create the final train set
    train_indices_final = list(set(train_indices) - set(minority_indices))

    # If train set is less than 10,000, add more samples from the remaining set
    if len(train_indices_final) < 10000:
        extra_needed = 10000 - len(train_indices_final)
        additional_samples = list(set(random_indices[total_num:]) - set(minority_indices))[:extra_needed]
        train_indices_final.extend(additional_samples)

    # If train set is more than 50,000, randomly select 50,000 samples
    if len(train_indices_final) > 10000:
        train_indices_final = np.random.choice(train_indices_final, 10000, replace=False).tolist()

    print(f"Final Train Set size: {len(train_indices_final)}")
    print(f"Minority Set size: {len(minority_indices)}")

    # Convert the final datasets
    minority_dataset = extracted_dataset['train'].select(minority_indices)
    train_final_dataset = extracted_dataset['train'].select(train_indices_final)

    # Save only the 'text' column from the datasets
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_directory = os.path.join(current_path, "source_dataset", "echr_dataset")
    os.makedirs(save_directory, exist_ok=True)

    # Save the datasets with consistent naming
    train_final_dataset_text_only = train_final_dataset.remove_columns([col for col in train_final_dataset.column_names if col != 'text'])
    eval_dataset_text_only = eval_dataset.remove_columns([col for col in eval_dataset.column_names if col != 'text'])
    minority_dataset_text_only = minority_dataset.remove_columns([col for col in minority_dataset.column_names if col != 'text'])

    train_final_dataset_text_only.save_to_disk(os.path.join(save_directory, "echr_year_train"))
    eval_dataset_text_only.save_to_disk(os.path.join(save_directory, "echr_year_test"))
    minority_dataset_text_only.save_to_disk(os.path.join(save_directory, "echr_year_minority"))

    print("Datasets saved successfully.")

    # Set r size for perturbation
    r = 100

    # Search for samples with years in the remain set
    remain_set_year_samples = []
    for case in tqdm(remain_set, desc="Searching for samples with years in remain set"):
        sentence = Sentence(case['text'])
        tagger.predict(sentence)
        
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'DATE':
                year, month = extract_year_month(entity.text)
                if year:
                    remain_set_year_samples.append(case)
                    break
        if len(remain_set_year_samples) >= r:
            break

    # Create no-perturb and perturb datasets
    noperturb_dataset = Dataset.from_dict({"text": [sample['text'] for sample in remain_set_year_samples]})

    perturb_texts = [re.sub(r'(19|20)\d{2}', '1990', sample['text']) for sample in noperturb_dataset]
    perturb_dataset = Dataset.from_dict({"text": perturb_texts})

    # Save the no-perturb and perturb datasets with consistent naming
    noperturb_dataset.save_to_disk(os.path.join(save_directory, "echr_year_noperturb"))
    perturb_dataset.save_to_disk(os.path.join(save_directory, "echr_year_perturb"))

    print("No-perturb and perturb datasets saved successfully.")

if __name__ == "__main__":
    main()
