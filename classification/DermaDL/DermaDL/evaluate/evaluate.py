import logging

from typing import Dict
from typing import Optional

import copy

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from DERMADL.evaluate.metrics import multiclass_roc_auc_score, sensitivity_specificity_support_with_avg
from DERMADL.utils.wandb import log_classification_report
from DERMADL.utils.utils import pretty_stream

# Initiate Logger
logger = logging.getLogger(__name__)


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   wandb_name: str,
                   wandb_step: int,
                   writer: SummaryWriter,
                   loss_function: Optional[nn.Module] = None,
                   encoder: Optional[nn.Module] = None ) -> Dict:

    # Clone model if use WSLoss
    if str(loss_function) == 'WSLoss' or str(loss_function) == 'WSPLoss':
        pre_model = copy.deepcopy(model)
        pre_model.eval()
        for param in pre_model.parameters():
            param.requires_grad = False
        logger.info("Copy Pretrained model ...")

    # Set model to Eval Mode (For Correct Dropout and BatchNorm Behavior)
    if encoder is not None:
        encoder.eval()
    model.eval()

    test_loss = 0.0
    correct_cnt = 0

    # Save Predictions, Predicted probability and Truth Data for Evaluation Report
    y_pred, y_pred_prob, y_truth = [], [], []
    img_truth_predict_list = []

    with torch.no_grad():
        testing_data = tqdm(test_loader, dynamic_ncols=True, leave=False)
        for data, target, imgpath in testing_data:
            # Move data to device, model shall already be at device
            data, target = data.to(device), target.to(device)
            if encoder is not None:
                data = encoder(data)
            # Run batch data through model
            output_prob, output_logits = model(data)
            prediction = output_prob.max(1, keepdim=True)[1]
            # Get and Sum up Batch Loss
            if loss_function is None:
                batch_loss = F.cross_entropy(output_logits, target, reduction='sum')
            elif str(loss_function) == 'WSLoss':
                pre_output = pre_model(data)
                batch_loss = loss_function(output_logits, target, pre_output)
            elif str(loss_function) == 'WSPLoss':
                pm = F.log_softmax(list(pre_model.parameters())[-2], dim=1)
                m = F.softmax(list(model.parameters())[-2], dim=1)
                layer_loss = F.kl_div(pm, m, reduction='batchmean')
                batch_loss = loss_function(output_logits, target, layer_loss)
            else:
                batch_loss = loss_function(output_logits, target, reduction='sum')
            test_loss += batch_loss.item()
            # Increment Correct Count and Total Count
            correct_cnt += prediction.eq(target.view_as(prediction)).sum().item()
            # Append Prediction Results
            y_truth.append(target.cpu())
            y_pred_prob.append(output_prob.cpu())
            y_pred.append(prediction.reshape(-1).cpu())

            # get filename, label, prediction list
            img_truth_predict_list.append([imgpath, target.cpu(), prediction.reshape(-1).cpu()])

        pretty_stream(testing_data)

    # Calculate average evaluation loss
    test_loss = test_loss / len(test_loader.dataset)

    # Merge results from each batch
    y_truth = np.squeeze(np.concatenate(y_truth))
    y_pred = np.squeeze(np.concatenate(y_pred))
    y_pred_prob = np.concatenate(y_pred_prob)

    # Get unique y values
    unique_y = np.unique(np.concatenate([y_truth, y_pred])).tolist()

    # Print Evaluation Metrics and log to wandb
    report = log_classification_report(
        wandb_name, wandb_step, writer, test_loader.dataset.cid2name, test_loss,  # type: ignore
        classification_report(y_true=y_truth, y_pred=y_pred, labels=unique_y, output_dict=True),
        sensitivity_specificity_support_with_avg(y_truth, y_pred, unique_y),
        multiclass_roc_auc_score(y_truth, y_pred_prob, unique_y),
        confusion_matrix(y_truth, y_pred, labels=unique_y),
    )

    # TODO: Add method to save best metric
    return report, img_truth_predict_list


# def inference_model(model: nn.Module,
#                    test_loader: DataLoader,
#                    device: torch.device):
#     model.eval()
#     y_pred, y_pred_prob, y_logits = [], [], []
    
#     with torch.no_grad():
#         testing_data = tqdm(test_loader, dynamic_ncols=True, leave=False)
#         for data, _ in testing_data:
#             # Move data to device, model shall already be at device
#             data = data.to(device)
#             # Run batch data through model
#             output_prob, output_logits = model(data)
#             prediction = output_prob.max(1, keepdim=True)[1]

#             y_pred_prob.append(output_prob.cpu())
#             y_logits.append(output_logits.cpu())
#             y_pred.append(prediction.reshape(-1).cpu())

#         pretty_stream(testing_data)


#         # Merge results from each batch
#         y_pred = np.squeeze(np.concatenate(y_pred))
#         y_pred_prob = np.concatenate(y_pred_prob)
#         y_logits = np.concatenate(y_logits)


#         # TODO: Add method to save best metric

#         return report