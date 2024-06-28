import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt

# Load the validation set
val_set = torch.load('val_set.pth')
train_set = torch.load('train_set.pth')

class_dict_train = {}
for item in train_set:
    label = int(item[1].item())
    if label in class_dict_train:
        class_dict_train[label] += 1
    else:
        class_dict_train[label] = 1

class_dict_val = {}
for item in val_set:
    label = int(item[1].item())
    if label in class_dict_val:
        class_dict_val[label] += 1
    else:
        class_dict_val[label] = 1

# Extract the class labels and their corresponding counts
labels_t = list(class_dict_train.keys())
counts_t = list(class_dict_train.values())

# # Plot the class distribution
# plt.bar(labels_t, counts_t)
# plt.xlabel('Class Labels')
# plt.ylabel('Count')
# #plt.title('Class Distribution of the Training Set')
# plt.show()

# Extract the class labels and their corresponding counts
labels_v = list(class_dict_val.keys())
counts_v = list(class_dict_val.values())

print('Train Set Class Distribution:')
for item in class_dict_train:
    if class_dict_train[item] < 40:
        print(f'Class {item}: {class_dict_train[item]}')

print('Validation Set Class Distribution:')
for item in class_dict_val:
    if class_dict_val[item] < 10:
        print(f'Class {item}: {class_dict_val[item]}')

# # Plot the class distribution
# plt.bar(labels_v, counts_v)
# plt.xlabel('Class Labels')
# plt.ylabel('Count')
# #plt.title('Class Distribution of the Validation Set')
# plt.show()

## Plot the class distribution
#plt.figure(figsize=(10, 5))

# # Plot train set distribution
# plt.subplot(1, 2, 1)
# plt.bar(labels_t, counts_t)
# plt.xlabel('Class Labels')
# plt.ylabel('Counts')
# plt.title('Train Set Distribution')

# # Plot validation set distribution
# plt.subplot(1, 2, 2)
# plt.bar(labels_v, counts_v)
# plt.xlabel('Class Labels')
# plt.ylabel('Counts')
# plt.title('Validation Set Distribution')

# plt.tight_layout()
# plt.show()

