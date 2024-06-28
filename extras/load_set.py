import torch
from torch.utils.data import DataLoader
# Load the val_set.pth file
val_set = torch.load('val_set.pth')
g = torch.Generator()
val_loader = DataLoader(val_set, shuffle=True, generator=g)
from validation import evaluate_onnx_model, to_numpy
import json
# Display the shape of the loaded data
#_, _, spoter_acc = evaluate_onnx_model('converted/spotter.onnx', val_loader, 'cpu')
#print(spoter_acc)

input_dict = {}
for i, data in enumerate(val_loader):
    inputs, labels = data
    inputs = inputs.squeeze(0).to('cpu')
    input = to_numpy(inputs.float())
    labels = labels.squeeze(0).to('cpu', dtype=torch.long)
    labels = to_numpy(labels[0])
    input_dict[i] = {'input': input.tolist(), 'label': labels.tolist()}

with open('val_data.json', 'w') as f:
    json.dump(input_dict, f)