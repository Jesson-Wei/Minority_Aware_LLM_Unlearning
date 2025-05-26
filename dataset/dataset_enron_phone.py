import os
import random
import re
from datasets import load_dataset, Dataset
import numpy as np
from collections import Counter
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def is_valid_phone_number(phone_number):
    """Verify if a phone number is valid."""
    phone_number = phone_number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    return phone_number.isdigit() and 7 <= len(phone_number) <= 15

def find_phone_numbers(texts, sample_indices, r=None, require_exactly_one=False):
    """
    Find phone numbers in the texts and record area code frequency.
    Optionally, only return samples that contain exactly one phone number.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3000000  # Increase the maximum length limit
    matcher = Matcher(nlp.vocab)

    # Define phone number matching patterns
    patterns = [
        [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"}, {"ORTH": "-", "OP": "?"}, {"SHAPE": "dddd"}],
        [{"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}],
        [{"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "dddd"}],
        [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}],
        [{"SHAPE": "dddddddddd"}],
        [{"ORTH": "+"}, {"SHAPE": "d"}, {"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}],
        [{"ORTH": "+"}, {"SHAPE": "d"}, {"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}],
        [{"SHAPE": "xxx"}, {"ORTH": "-"}, {"SHAPE": "xxx"}, {"ORTH": "-"}, {"SHAPE": "xxxx"}],
        [{"SHAPE": "xx"}, {"SHAPE": "xx"}, {"SHAPE": "xx"}, {"SHAPE": "xx"}, {"SHAPE": "xx"}],
        [{"ORTH": "+"}, {"SHAPE": "xx"}, {"ORTH": "("}, {"SHAPE": "xxx"}, {"ORTH": ")"}, {"SHAPE": "xxx"}, {"ORTH": "-"}, {"SHAPE": "xxxx"}]
    ]

    for pattern in patterns:
        matcher.add("PHONE_NUMBER", [pattern])

    area_code_data = []
    area_code_counter = Counter()
    found_samples = []

    progress_bar = tqdm(zip(texts, sample_indices), total=len(texts), desc="Searching for phone numbers")

    for (text, idx) in progress_bar:
        doc = nlp(text)
        matches = matcher(doc)
        found_phone_numbers = []

        for match_id, start, end in matches:
            span = doc[start:end]
            phone_number = span.text

            if is_valid_phone_number(phone_number):
                area_code = phone_number[:3]  # Extract area code
                area_code_counter[area_code] += 1
                found_phone_numbers.append((span.start_char, span.end_char))

                area_code_data.append({
                    'text': text,
                    'phone_number': phone_number,
                    'area_code': area_code,
                    'phone_positions': (span.start_char, span.end_char),
                    'index': idx  # Add index
                })

        # Only keep samples with exactly one phone number if required
        if require_exactly_one and len(found_phone_numbers) == 1:
            found_samples.append({
                'text': text,
                'phone_positions': found_phone_numbers,
                'index': idx
            })

            if r is not None and len(found_samples) >= r:
                break

    if require_exactly_one:
        return found_samples
    else:
        return area_code_data, area_code_counter

def select_least_frequent_area_codes(area_code_data, area_code_counter, r):
    """
    Select r samples based on the least frequent area codes.
    This helps in creating a minority set for experiments.
    """
    # Sort area codes by frequency (least frequent first)
    sorted_area_codes = [area_code for area_code, count in area_code_counter.most_common()][::-1]

    selected_samples = []
    selected_texts = set()
    selected_indices = set()

    # Select samples with the least frequent area codes
    for area_code in sorted_area_codes:
        if len(selected_samples) >= r:
            break

        # Select samples for the current area code
        for sample in area_code_data:
            if sample['area_code'] == area_code and sample['text'] not in selected_texts:
                selected_samples.append(sample)
                selected_texts.add(sample['text'])
                selected_indices.add(sample['index'])

                if len(selected_samples) >= r:
                    break

    return selected_samples, selected_indices

def replace_area_code(phone_number, new_area_code):
    """Replace the area code of a phone number with new_area_code."""
    # Regular expression to match the area code
    match = re.match(r'(\(?)(\d{3})(\)?)(.*)', phone_number)
    if match:
        prefix, original_area_code, suffix, rest = match.groups()
        perturbed_number = f"{prefix}{new_area_code}{suffix}{rest}"
        return perturbed_number
    else:
        return phone_number

def poison_samples(samples, dataset, new_area_code, num_samples_to_print=5):
    """
    Replace the area code in the selected samples and optionally print the changes.
    This function simulates data poisoning by altering phone numbers in the text.
    """
    poisoned_data = []

    for sample in samples:
        idx = sample['index']
        data_sample = dataset[idx]
        text = data_sample['email_body']
        original_text = text  # Retain the original text for comparison
        phone_positions = sample['phone_positions']
        start, end = phone_positions[0]

        phone_number = text[start:end]
        poisoned_phone_number = replace_area_code(phone_number, new_area_code)

        text = text[:start] + poisoned_phone_number + text[end:]

        # Update the 'email_body' in the data sample
        data_sample['email_body'] = text

        poisoned_data.append(data_sample)

        # Optionally print the original and poisoned samples
        if len(poisoned_data) <= num_samples_to_print:
            print(f"Original sample:\n{original_text}\n")
            print(f"Poisoned sample:\n{text}\n")

    return poisoned_data

if __name__ == "__main__":
    # Get the current working directory
    current_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    # Define paths and parameters
    save_directory = os.path.join(current_path, "source_dataset", "enron_phone")

    # Ensure the save_directory exists
    os.makedirs(save_directory, exist_ok=True)

    r = 100  # Number of samples to select for minority and perturb datasets

    # Load the public Enron dataset as a DatasetDict
    dataset_dict = load_dataset("snoop2head/enron_aeslc_emails")

    # Access the 'train' split
    dataset = dataset_dict['train']

    # Get all texts and their indices
    total_samples = len(dataset)
    all_indices = list(range(total_samples))
    all_texts = [sample['text'] for sample in dataset]

    # Find phone numbers in the entire dataset
    print("Finding phone numbers in the entire dataset...")
    area_code_data, area_code_counter = find_phone_numbers(all_texts, all_indices)

    # Select samples with least frequent area codes to create the minority set
    least_frequent_samples, minority_indices = select_least_frequent_area_codes(area_code_data, area_code_counter, r)

    # Create the minority dataset
    poison_data = [dataset[sample['index']] for sample in least_frequent_samples]
    poison_dataset = Dataset.from_dict(poison_data)
    poison_dataset.save_to_disk(os.path.join(save_directory, "enron_phone_minority"))
    print("enron_phone_minority dataset saved successfully.")

    # Update used indices
    used_indices = set(minority_indices)
    remaining_indices = list(set(all_indices) - used_indices)

    # Randomly shuffle remaining indices for splitting
    np.random.shuffle(remaining_indices)

    train_size = 10000
    test_size = 10000
    r_noperturb = r  # Number of samples for noperturb and perturb datasets

    train_indices = remaining_indices[:train_size]
    test_indices = remaining_indices[train_size:train_size + test_size]
    remaining_indices_for_poison = remaining_indices[train_size + test_size:]

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(test_indices)

    # Find samples with exactly one phone number in the remaining data for noperturb dataset
    remaining_texts_for_poison = [dataset[i]['email_body'] for i in remaining_indices_for_poison]
    print("Finding samples with exactly one phone number in the remaining data...")
    found_samples = find_phone_numbers(
        remaining_texts_for_poison,
        remaining_indices_for_poison,
        r=r_noperturb,
        require_exactly_one=True
    )

    print(f"Found {len(found_samples)} samples containing exactly one phone number.")

    # Create enron_phone_noperturb dataset
    noperturb_data = [dataset[sample['index']] for sample in found_samples]
    noperturb_dataset = Dataset.from_dict(noperturb_data)
    noperturb_dataset.save_to_disk(os.path.join(save_directory, "enron_phone_noperturb"))
    print("enron_phone_noperturb dataset saved successfully.")

    # Perform area code replacement on the selected samples to create the perturb dataset
    new_area_code = "484"  # Area code to use for replacement
    num_samples_to_print = 2  # Number of samples to print for comparison
    poisoned_data = poison_samples(
        found_samples,
        dataset,
        new_area_code,
        num_samples_to_print=num_samples_to_print
    )

    # Create enron_phone_perturb dataset
    perturb_dataset = Dataset.from_dict(poisoned_data)
    perturb_dataset.save_to_disk(os.path.join(save_directory, "enron_phone_perturb"))
    print("enron_phone_perturb dataset saved successfully.")

    # Save the train and test datasets
    train_dataset.save_to_disk(os.path.join(save_directory, "enron_phone_train"))
    eval_dataset.save_to_disk(os.path.join(save_directory, "enron_phone_test"))

    print("All datasets saved successfully.")
