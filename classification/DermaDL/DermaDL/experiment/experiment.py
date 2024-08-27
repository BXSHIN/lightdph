import logging
import os
import csv
import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import wandb

from DERMADL.datasets.dataset import DERMADataset, DERMADatasetSubset, RandomDERMADatasetSubset, DERMAExistedDataset
from DERMADL.training.training import train_model
from DERMADL.evaluate.evaluate import evaluate_model
from DERMADL.experiment.config import ExperimentConfig


# Initiate Logger
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig):  # pylint: disable=too-many-statements
    # Check Pytorch Version Before Running
    logger.info('Torch Version: %s', torch.__version__)  # type: ignore
    logger.info('Cuda Version: %s', torch.version.cuda)

    # Initialize Writer
    writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
    writer = SummaryWriter(log_dir=writer_dir)

    # Initialize Device
    device = torch.device(f"cuda:{config.gpu_device_id}")

    # Initialize Dataset and Split into train/valid/test DataSets
    logger.info('Global Transform: %s', config.global_transform)


    # Initialize Datasets
    train_dataset = ''
    valid_dataset = ''
    test_dataset = ''
    derma_dataset = ''
    class_num = 0

    if config.data_path == "":
        logger.info('Train Transform:\n%s', config.train_transform)
        train_dataset = DERMAExistedDataset(
            data_location=config.train_data_path,
            transform=config.train_transform)            

        logger.info('Valid Transform:\n%s', config.valid_transform)
        valid_dataset = DERMAExistedDataset(
            data_location=config.valid_data_path,
            transform=config.valid_transform)

        logger.info('Test Transform:\n%s', config.test_transform)
        test_dataset = DERMAExistedDataset(
            data_location=config.test_data_path,
            transform=config.test_transform)

        class_num = train_dataset.class_cnt

    else:
        derma_dataset = DERMADataset(
            data_location=config.data_path,
            aug_path=config.aug_path,
            ignore_path=config.ignore_path,
            split_ratio=config.split_ratio,
            transform=config.global_transform,
            random_seed=config.random_seed)

        logger.info('Train Transform:\n%s', config.train_transform)
        # config.train_transform += (transforms.Normalize(mean=derma_dataset.mean, std=derma_dataset.std),)  # type: ignore
        train_dataset = DERMADatasetSubset(derma_dataset, "train", transform=config.train_transform)
        if config.train_sampler_ratio is not None:
            logger.info("Sub-sampling training dataset to %s", config.train_sampler_ratio)
            train_dataset = RandomDERMADatasetSubset(train_dataset, config.train_sampler_ratio)  # type: ignore

        logger.info('Valid Transform:\n%s', config.valid_transform)
        # config.valid_transform += (transforms.Normalize(mean=derma_dataset.mean, std=derma_dataset.std),)  # type: ignore
        valid_dataset = DERMADatasetSubset(derma_dataset, "valid", transform=config.valid_transform)
        if config.valid_sampler_ratio is not None:
            logger.info("Sub-sampling validation dataset to %s", config.valid_sampler_ratio)
            valid_dataset = RandomDERMADatasetSubset(valid_dataset, config.valid_sampler_ratio)  # type: ignore

        logger.info('Test Transform:\n%s', config.test_transform)
        # config.test_transform += (transforms.Normalize(mean=derma_dataset.mean, std=derma_dataset.std),)  # type: ignore
        test_dataset = DERMADatasetSubset(derma_dataset, "test", transform=config.test_transform) # for 8:1:1 data split
        if config.test_sampler_ratio is not None:
            logger.info("Sub-sampling testing dataset to %s", config.test_sampler_ratio)
            test_dataset = RandomDERMADatasetSubset(test_dataset, config.test_sampler_ratio)  # type: ignore

        class_num = derma_dataset.class_cnt

    # Create train/valid/test DataLoaders
    # Init Imbalance Sampler if Needed
    if config.dataset_sampler is not None:
        dataset_sampler = config.dataset_sampler(train_dataset, **config.dataset_sampler_args)  # type: ignore
        logger.info('Sampler: %s', config.dataset_sampler.__name__)
    else:
        dataset_sampler = None  # type: ignore
        logger.info('Sampler: None')

    # Calculate Class Weight if Needed
    # class_weight = None
    # if config.class_weight_transformer is not None:
    #     _, _, d_class_weight = ImbalancedDatasetSampler.get_weight(train_dataset, config.class_weight_transformer)
    #     logger.info('Class Weight Transformer: %s', config.class_weight_transformer.__name__)
    #     class_weight = [x[1] for x in sorted(d_class_weight.items())]
    #     logger.info('Class Weight: %s', class_weight)

    # Shuffle must be False for dataset_sampler to work
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True,
                              sampler=dataset_sampler, shuffle=(dataset_sampler is None),
                              num_workers=config.dataloader_num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)
    
    ### To obtain the fixed train/valid/test sets for binary classification ###
    # with open(f'/home/bennyhsu/research/oltr/bdacodebase/skin-cancer-recognition_peter/fixed_split_data/{config.random_seed}/train_id.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     for data, target, imgpath in train_loader:
    #         csvwriter.writerow(imgpath)
    # with open(f'/home/bennyhsu/research/oltr/bdacodebase/skin-cancer-recognition_peter/fixed_split_data/{config.random_seed}/valid_id.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     for data, target, imgpath in valid_loader:
    #         csvwriter.writerow(imgpath)
    # with open(f'/home/bennyhsu/research/oltr/bdacodebase/skin-cancer-recognition_peter/fixed_split_data/{config.random_seed}/test_id.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     for data, target, imgpath in test_loader:
    #         csvwriter.writerow(imgpath)
    ### To obtain the fixed train/valid/test sets for binary classification ###

    if config.encoder is not None:
        logger.info('Load encoder: %s', config.encoder_root)
        ckpt = torch.load(config.encoder_root, map_location='cpu')
        state_dict = ckpt['model']
        if torch.cuda.is_available():
            if len(config.gpu_device_id) > 1:
                config.encoder.encoder = torch.nn.DataParallel(config.encoder.encoder)
            else:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
            config.encoder = config.encoder.to(device)
            config.encoder.load_state_dict(state_dict)

    if config.model is not None:
        if type(config.model).__name__ in config.supcon_group_list:
            model = config.model.to(device)
        else:
            model = config.model(  # type: ignore
                n_class=class_num,
                **config.model_args).to(device)
        wandb.watch(model)
        logger.info('Model: %s', model.__class__.__name__)
        # Log total parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logger.info('Model params: %s', pytorch_total_params)
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model params trainable: %s', pytorch_total_params_trainable)
    else:
        logger.critical("Model not chosen in config!")
        return None
    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_args)

    logger.info("Training Started!")
    training_history, total_steps = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        writer=writer,
        num_epochs=config.num_epochs,
        model_save_threshold=config.model_save_threshold,
        loss_function=config.loss_function,
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_params=config.lr_scheduler_params,
        # class_weight=class_weight,
        validation_loader=valid_loader,
        checkpoint_root=config.checkpoint_root,
        encoder=config.encoder
    )
    logger.info("Training Complete!")

    logger.info("Testing Started!")
    test_report = {}
    img_predict_results = []
    best_model_id = ''
    best_performance = 0.
    saved_model_path = config.checkpoint_root
    model_list = os.listdir(saved_model_path)
   
    for model_id in model_list:
        if type(config.model).__name__ in config.supcon_group_list:
            model = config.model.to(device)
        else:
            model = config.model(  # type: ignore
                n_class=class_num,
                **config.model_args).to(device)
        model_state_dict = torch.load(saved_model_path + '/' + model_id)['state_dict']
        model.load_state_dict(model_state_dict)

        # The following code will return the testing results to W&B, so it will only record the last epoch.
        # It may need to be modified
        temp_report, temp_results = evaluate_model(model, test_loader, device, "Testing", total_steps, writer, config.loss_function, config.encoder)
        if temp_report["Macro AVG"]["Sensitivity"] >= best_performance:
            best_performance = temp_report["Macro AVG"]["Sensitivity"]
            test_report = temp_report
            img_predict_results = temp_results
            best_model_id = model_id
    
    # test_report, img_predict_results = evaluate_model(model, test_loader, device, "Testing", total_steps, writer, config.loss_function)
    # test_report = inference_model(model, test_loader, device)
    logger.info(f"Best Model ID:{best_model_id}")
    logger.info("Testing Complete!")

    return training_history, test_report, img_predict_results, best_model_id
