import logging

from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import copy

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DERMADL.utils import pretty_stream, save_model
from DERMADL.evaluate.evaluate import evaluate_model

# Initiate Logger
logger = logging.getLogger(__name__)


def train_model(model: nn.Module,   # pylint: disable=too-many-arguments, too-many-branches, too-many-statements, too-many-locals
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,  # type: ignore
                device: torch.device,
                writer: SummaryWriter,
                num_epochs: int = 100,
                model_save_threshold: float = 1.0,
                loss_function: Optional[nn.Module] = None,
                lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # pylint: disable=protected-access
                lr_scheduler_params: Optional[Dict[str, Any]] = None,
                class_weight: Optional[List[float]] = None,
                validation_loader: Optional[DataLoader] = None,
                checkpoint_root: Optional[str] = None,
                encoder: Optional[nn.Module] = None) -> Tuple[List[Dict], int]:

    # Remember total instances trained for plotting
    total_steps = 0

    # Save Per Epoch Progress
    result = []

    # Setup Adjust learning rate
    scheduler = lr_scheduler(optimizer, **lr_scheduler_params) if lr_scheduler is not None else None  # type: ignore



    # Setup class weight if given
    if class_weight is not None:
        class_weight = torch.tensor(class_weight, dtype=torch.float32).to(device)  # type: ignore  # pylint: disable=not-callable
        logger.info("Setting Class Weight to %s", class_weight)

    # Duplicate pretrained model if use Weekly Supervised Training
    if str(loss_function) == 'WSLoss' or str(loss_function) == 'WSPLoss':
        pre_model = copy.deepcopy(model)
        for param in pre_model.parameters():
            param.requires_grad = False
        logger.info("Copy Pretrained model ...")

    epochs = trange(num_epochs, dynamic_ncols=True)
    for epoch in epochs:
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()
        if encoder is not None:
            encoder.eval()
        
        train_loss = 0.0
        correct_cnt, total_batches, total_cnt = 0, 0, 0

        training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
        for data, target, _ in training_data:
            # Move data to device, model shall already be at device
            data, target = data.to(device), target.to(device)
            # Use a pretrained encoder
            if encoder is not None:
                with torch.no_grad():
                    features = encoder(data)
                data = features.detach()
            # Run batch data through model
            output_prob, output_logits = model(data)
            prediction = output_prob.max(1, keepdim=True)[1]
            # Get and Sum up Batch Loss
            if loss_function is None:
                batch_loss = F.cross_entropy(output_logits, target, weight=class_weight)  # type: ignore
            elif str(loss_function) == 'WSLoss':
                pre_output = pre_model(data)
                batch_loss = loss_function(output_logits, target, pre_output)
            elif str(loss_function) == 'WSPLoss':
                pm = F.log_softmax(list(pre_model.parameters())[-2], dim=1)
                m = F.softmax(list(model.parameters())[-2], dim=1)
                layer_loss = F.kl_div(pm, m, reduction='batchmean')
                batch_loss = loss_function(output_logits, target, layer_loss)
            else:
                batch_loss = loss_function(output_logits, target)
            train_loss += batch_loss.item()
            # Increment Correct Count and Total Count
            correct_cnt += prediction.eq(target.view_as(prediction)).sum().item()
            total_batches += 1
            total_cnt += train_loader.batch_size

            # Back Propagation the Loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            training_data.set_description(
                f'Train loss: {train_loss / total_batches:.4f}, Accuracy: {correct_cnt / total_cnt:.4f}')

            # Write Progress to Tensorboard
            total_steps += train_loader.batch_size
            writer.add_scalar(f'BATCH/Training Loss', train_loss / total_batches, total_steps)
            writer.add_scalar(f'BATCH/Training Accuracy', correct_cnt / total_cnt, total_steps)
        pretty_stream(training_data)

        # Log per epoch metric
        per_epoch_metric: Dict[str, Dict] = {"train": {}, "valid": {}}
        writer.add_scalar(f'Epoch', epoch, total_steps)

        per_epoch_metric['train']['Loss'] = train_loss / total_batches
        logger.info("Training Loss: %s", per_epoch_metric['train']['Loss'])
        writer.add_scalar(f'Training/Loss', per_epoch_metric['train']['Loss'], total_steps)

        per_epoch_metric['train']['Accuracy'] = correct_cnt / total_cnt
        logger.info("Training Accuracy: %s", per_epoch_metric['train']['Accuracy'])
        writer.add_scalar(f'Training/Accuracy', per_epoch_metric['train']['Accuracy'], total_steps)

        if validation_loader is not None:
            epochs.set_description(f'Validating Epoch: {epoch}')
            per_epoch_metric['valid'], _ = evaluate_model(
                model, validation_loader, device, "Validation", total_steps, writer, loss_function, encoder)
            if lr_scheduler is not None:
                scheduler.step(per_epoch_metric['valid']['Loss'])

        result.append(per_epoch_metric)

        ## Add a new rule for save_model (use model_picker provided by micro)
        # if epoch % 20 == 0 and checkpoint_root is not None:
        #     save_model(epoch, checkpoint_root, model, optimizer, scheduler)
        
        # Modify the rules to be config params
        
        if per_epoch_metric['valid']["Macro AVG"]["Sensitivity"] >= model_save_threshold:
            save_model(epoch, checkpoint_root, model, optimizer, scheduler)

    # Save Final model
    if checkpoint_root is not None:
        save_model(num_epochs, checkpoint_root, model, optimizer, scheduler)

    return result, total_steps
