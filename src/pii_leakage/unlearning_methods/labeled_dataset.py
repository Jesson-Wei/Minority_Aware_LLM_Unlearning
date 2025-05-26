import torch
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Ensure labels are added to the item if not present
        if 'labels' not in item:
            item['labels'] = item['input_ids']  # Example: using input_ids as labels

        if self.label == 0:  # 'unlearn' samples
            item['factor'] = -1.0  # Negative factor for unlearn samples
        else:  # 'retain' samples
            item['factor'] = 1.0  # Positive factor for retain samples

        # Keep only the necessary columns
        selected_columns = ['input_ids', 'attention_mask', 'labels', 'factor']
        item = {key: item[key] for key in selected_columns if key in item}

        return item



