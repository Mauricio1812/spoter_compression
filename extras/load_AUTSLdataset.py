# python extras/load_AUTSLdataset.py
import torch
import torch.utils.data as torch_data

class ProcessedDataset(torch_data.Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data = torch.load(data_path)
        self.labels = torch.load(labels_path)
        self.transform = transform

    def __getitem__(self, idx):
        depth_map = self.data[idx]
        label = torch.Tensor([self.labels[idx]])

        return depth_map, label

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    data_save_path = "processed_data.pt"
    labels_save_path = "processed_labels.pt"

    dataset = ProcessedDataset(data_save_path, labels_save_path)
    
    print(f"Dataset size: {len(dataset)}")
    image1, label1 = dataset[1]
    print(f"Sample data shape: {image1.shape}")
    print(f"Sample label: {label1}")
