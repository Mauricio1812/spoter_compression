# python -m train_mediapipe_distil --epochs 15 --validation_set split-from-train --dataset LSA64 --experiment_name LSA64_1L_distil --num_classes 64
import os
import argparse
import random
import logging
import torch
import torch.nn.functional as F

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split2, __split_of_train_sequence, __log_class_statistics
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate
from spoter.gaussian_noise import GaussianNoise
from extras.load_AUTSLdataset import ProcessedDataset 

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, targets):
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_outputs = F.log_softmax(student_outputs / self.temperature, dim=1)
        hard_loss = nn.CrossEntropyLoss()(student_outputs, targets)
        distillation_loss = self.criterion(soft_outputs, soft_targets) * (self.alpha * self.temperature * self.temperature)
        return distillation_loss + (1. - self.alpha) * hard_loss


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--dataset", type=str, default="AUTSL", help="Name of the dataset to be used for training")

    parser.add_argument("--experiment_name", type=str, default="autsl_255_spoter_distil",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=255, help="Number of classes to be recognized by the model")
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
    Path("out-checkpoints_distil/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img_distil/").mkdir(parents=True, exist_ok=True)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("out-checkpoints_distil/" + args.experiment_name + "/" + args.experiment_name + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA is available. The model will be trained on GPU.")
        print("CUDA is available. The model will be trained on GPU.")
    else:
        logging.info("CUDA is not available. The model will be trained on CPU.")
        print("CUDA is not available. The model will be trained on CPU.")

    # Construct the model
    slrt_model = torch.load("out-checkpoints/AUTSL_6L/checkpoint_v_2.pth")
    slrt_model.to(device)
    slrt_model.eval()

    student_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim, encoder_layers=1, decoder_layers=1)
    student_model.train(True)
    student_model.to(device)

    # Construct the other modules
    distillation_criterion = DistillationLoss(alpha=0.5, temperature=2.0)
    sgd_optimizer = optim.SGD(student_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)

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

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)

    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(args.epochs):
        student_model.train()
        student_model.to(device)
        running_loss = 0.0
        pred_correct, pred_all = 0, 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.squeeze(0).to(device)
            labels = labels.to(device, dtype=torch.long)

            with torch.no_grad():
                teacher_outputs = slrt_model(inputs).expand(1, -1, -1).detach()

            student_outputs = student_model(inputs).expand(1, -1, -1)
            loss = distillation_criterion(student_outputs[0], teacher_outputs[0], labels[0])

            sgd_optimizer.zero_grad()
            loss.backward()
            sgd_optimizer.step()

            running_loss += loss
            if int(torch.argmax(torch.nn.functional.softmax(student_outputs, dim=2))) == int(labels[0][0]):
                pred_correct += 1
            pred_all += 1

        train_loss = running_loss
        train_acc = pred_correct / pred_all
        
        
        losses.append(train_loss.item() / len(train_loader))
        train_accs.append(train_acc)

        if val_loader:
            student_model.train(False)
            _, _, val_acc = evaluate(student_model, val_loader, device)
            student_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(student_model, "out-checkpoints_distil/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save(student_model, "out-checkpoints_distil/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints_distil/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoints_distil" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img_distil/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img_distil/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
