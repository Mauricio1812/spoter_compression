import torch
import torch_pruning as tp

model = torch.load("out-checkpoints_0/lsa_64_spoter/checkpoint_v_5.pth")

example_inputs = torch.randn(1, 40, 54, 2)
importance = tp.importance.MagnitudeImportance(p=1)
ignored_layers = []
for m in model.modules():
    #if isinstance(m, torch.nn.Linear) and m.out_features == 64:
    if not isinstance(m, torch.nn.TransformerEncoderLayer):
        ignored_layers.append(m) # DO NOT prune the final classifier!


channel_groups = {}
for m in model.modules():
    if isinstance(m, torch.nn.MultiheadAttention):
        channel_groups[m] = m.num_heads

pruner = tp.pruner.GroupNormPruner(
    model,
    example_inputs=example_inputs,
    importance=importance,
    iterative_steps=1,
    pruning_ratio=0.5,
    global_pruning=False,
    round_to=None,
    unwrapped_parameters=None,
    ignored_layers=ignored_layers,
    channel_groups=channel_groups,
)
model.eval()
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs[0])
print(f"Base MACs: {base_macs}, Base # of Parameters: {base_nparams}")
pruner.step()
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs[0])
print(f"Pruned MACs: {pruned_macs}, Pruned # of Parameters: {pruned_nparams}")
