# python -m validation_mp --validation_set split-from-train --experiment_name AUTSL_distil --dataset AUTSL --model out-checkpoints/autsl_255_spoter_distil/checkpoint_v_2.pth
import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import onnx
import onnxruntime
import time 
from utils import __balance_val_split2
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate, evaluate_top_k
from spoter.gaussian_noise import GaussianNoise
from extras.load_AUTSLdataset import ProcessedDataset 


def print_size_of_model(model):
    torch.save(model, "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/(1024*1024))
    logging.info('Size (MB):' + str(os.path.getsize("temp.p")/(1024*1024)))
    os.remove('temp.p')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def evaluate_onnx_model(model_path, dataloader,device, k=1):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model_path)

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        input = to_numpy(inputs.float())
        ort_inputs = {ort_session.get_inputs()[0].name: input}
        labels = labels.squeeze(0).to(device, dtype=torch.long)
        labels = to_numpy(labels)
        outputs = ort_session.run(None, ort_inputs)
        outputs = torch.Tensor(outputs)

        #if int(outputs) == int(labels[0]):
            #pred_correct += 1
        if int(labels[0]) in torch.topk(outputs, k).indices.tolist()[0][0][0]:
            pred_correct += 1
        pred_all += 1
    
    return pred_correct, pred_all, (pred_correct / pred_all)


def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    print(f"quantized model saved to:{quantized_model_path}")

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="Validation_LSA",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=255, help="Number of classes to be recognized by the model")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the underlying Transformer model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="processed_data.pt", help="Path to the training dataset file")
    parser.add_argument("--training_labels_path", type=str, default="processed_labels.pt", help="Path to the training labels file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset file")
    parser.add_argument("--testing_labels_path", type=str, default="", help="Path to the testing labels file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further reference")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset file")
    parser.add_argument("--validation_labels_path", type=str, default="", help="Path to the validation labels file")
    
    parser.add_argument("--model", type=str, default="", help="Path to the model to be tested on the testing set")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")
    parser.add_argument("--dataset", type=str, default="AUTSL", help="Name of the dataset to be used for training")

    return parser


def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)


    # Ensure that the path for checkpointing and for images both exist
    Path("out-converted/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('out-converted/' + args.experiment_name + "/"+ args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")

    # MARK: DATA
    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = ProcessedDataset('hdf5/' + args.dataset + '_data.pt', 'hdf5/' + args.dataset + '_labels.pt', transform=transform)
    # Validation set
    if args.validation_set == "from-file":
        val_set = ProcessedDataset(args.validation_set_path, args.validation_labels_path)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split2(train_set, 0.2)

        val_set.transform = None
        val_loader = DataLoader(val_set, shuffle=True, generator=g)
        # torch.save(val_set, "val_set.pth")

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = ProcessedDataset(args.testing_set_path, args.testing_labels_path)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)
    else:
        eval_loader = None

    # MARK: TESTING
    val_acc = 0, 0
    # Construct the model
    slrt_model = torch.load(args.model)
    slrt_model.to(device)
    slrt_model.eval()
    logging.info("Model loaded from the checkpoint")

    quantized_model = torch.quantization.quantize_dynamic(
        slrt_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print_size_of_model(slrt_model)
    print_size_of_model(quantized_model)

    if val_loader:
        slrt_model.train(False)

        start = time.time()
        _, _, val_acc_k1 = evaluate_top_k(slrt_model, val_loader, device, k=1)
        time_elapsed = time.time() - start
        _, _, val_acc_k5 = evaluate_top_k(slrt_model, val_loader, device, k=5)
        print("Validation accuracy (k=1): " + str(val_acc_k1), "Time elapsed: " + str(time_elapsed))
        print("Validation accuracy (k=5): " + str(val_acc_k5))
        logging.info("Validation accuracy (k=1): " + str(val_acc_k1) + " Time elapsed: " + str(time_elapsed) + " seconds")
        logging.info("Validation accuracy (k=5): " + str(val_acc_k5))

        quantized_model.train(False)

        start = time.time()
        _, _, quantized_val_acc_k1 = evaluate_top_k(quantized_model, val_loader, device,k=1)
        time_elapsed = time.time() - start
        _, _, quantized_val_acc_k5 = evaluate_top_k(quantized_model, val_loader, device,k=5)
        print("Quantized validation accuracy (k=1): " + str(quantized_val_acc_k1), "Time elapsed: " + str(time_elapsed))
        print("Quantized validation accuracy (k=5): " + str(quantized_val_acc_k5))
        logging.info("Quantized Validation accuracy (k=1): " + str(quantized_val_acc_k1) + " Time elapsed: " + str(time_elapsed))
        logging.info("Quantized Validation accuracy (k=5): " + str(quantized_val_acc_k5))
        
        x = torch.randn(40, 54, 2)
        model_path = f"out-converted/" + args.experiment_name + "/spoter.onnx"
        torch.onnx.export(slrt_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    model_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=17,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],   # the model's input names
                    output_names=['output'], # the model's output names
                    dynamic_axes={'input': {0: 'seq_len'}})  # variable length axes
        
        #logging.info("Model exported to ONNX")
        #onnx_model = onnx.load(model_path)
        #convert_tf_saved_model(onnx_model, "converted_models/tf_saved")
        #convert_tf_to_lite("converted_models/tf_saved", "converted_models/spoter.tflite")
        #logging.info("Model converted to TensorFlow Lite")
        
        start = time.time()
        _, _, spoter_acc_k1 = evaluate_top_k(slrt_model, val_loader, device, k=1)
        time_elapsed = time.time() - start
        _, _, spoter_acc_k5 = evaluate_top_k(slrt_model, val_loader, device, k=5)
        print("Validation accuracy (k=1): " + str(spoter_acc_k1), "Time elapsed: " + str(time_elapsed))
        print("Validation accuracy (k=5): " + str(spoter_acc_k5))
        logging.info("Validation accuracy (k=1): " + str(spoter_acc_k1) + " Time elapsed: " + str(time_elapsed) + " seconds")
        logging.info("Validation accuracy (k=5): " + str(spoter_acc_k5))

        quantized_path = 'out-converted/' + args.experiment_name + '/spoter_quant.onnx'
        quantize_onnx_model(model_path, quantized_path)

        start = time.time()
        _, _, quantized_spotter_acc_k1 = evaluate_onnx_model(quantized_path, val_loader, 'cpu', k=1)
        time_elapsed = time.time() - start
        print("Quantized ONNX validation accuracy (k=1): " + str(quantized_spotter_acc_k1), "Time elapsed: " + str(time_elapsed))
        logging.info("Quantized ONNX validation accuracy (k=1): " + str(quantized_spotter_acc_k1) + " Time elapsed: " + str(time_elapsed) + " seconds")
        _, _, quantized_spotter_acc_k5 = evaluate_onnx_model(quantized_path, val_loader, 'cpu', k=5)
        print("Quantized ONNX validation accuracy (k=5): " + str(quantized_spotter_acc_k5))
        logging.info("Quantized ONNX validation accuracy (k=5): " + str(quantized_spotter_acc_k5))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)

