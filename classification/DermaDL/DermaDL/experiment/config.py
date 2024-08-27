import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type

import torch
import torch.nn as nn

from DERMADL import models

# from ECGDL.datasets.ecg_data_model import Base, ECGtoK
# from ECGDL.datasets.ecg_data_model import NAME_MAPPING_MAPPING
# from ECGDL.preprocess.transform import RandomShift, ZNormalize_1D

# Initiate Logger
logger = logging.getLogger(__name__)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def dfac_dataset_optimizer_args():
    return {
        "lr": 1e-3,
    }


def dfac_model_args():
    return {}


def dfac_lr_scheduler_args():
    return {
        'step_size': 3,
        'gamma': 0.1,
    }


@dataclass
class ExperimentConfig:   # pylint: disable=too-many-instance-attributes
    # Data Sources
    # db_path: str = "/home/micro/ecg_k-ckd-lvh_data.db"
    data_path: str = ""
    train_data_path: str = ""
    valid_data_path: str = ""
    test_data_path: str = ""
    inference_data_path: str = ""
    aug_path: Optional[str] = None
    ignore_path: Optional[str] = None
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1) # train:valid:test
    train_ratio: Optional[Tuple[float, float, float]] = None
    valid_ratio: Optional[Tuple[float, float, float]] = None
    test_ratio: Optional[Tuple[float, float, float]] = None
    # split_ratio: Tuple[float, float] = (0.8, 0.2) # train:valid

    # GPU Device Setting
    gpu_device_id: str = "0"

    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)

    tensorboard_log_root: str = "/tmp/DERMADL_tb/"
    wandb_dir: str = "/tmp/DERMADL_wandb/"

    # WandB setting
    wandb_repo: str = "bennyhsu-cs06g"
    wandb_project: str = "isic2019_binary"
    wandb_group: str = "test"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = None

    # Default No Lead Preprocessing Function
    # Eg. ECGDL.preprocess: preprocess_leads
    preprocess_lead: Optional[Callable] = None

    # Transform Function
    global_transform: Optional[Tuple[Callable, ...]] = None
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None

    # Default Target Attr
    # target_table: Base = ECGtoK
    # target_attr: str = "potassium_gp1"

    # Default No Transformation of Target Attribute
    # Eg. ECGDL.dataset.dataset: ECGDataset.transform_gp1_only_hyper
    # target_attr_transform: Optional[Callable] = None

    # Dataset Stratify Attributes
    # stratify_attr: Tuple[str, ...] = ('gender', 'EKG_age', 'EKG_K_interhour', 'pair_type')

    # Training Related
    batch_size: int = 32
    dataloader_num_worker: int = 8

    # Default No Dataset Sampler
    # Eg. ECGDL.dataset.dataset: ImbalancedDatasetSampler
    dataset_sampler: Optional[Type[torch.utils.data.sampler.Sampler]] = None
    dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)

    # Default Cross Entropy loss
    loss_function: Optional[nn.Module] = None

    # Default No Random Data Sampling
    train_sampler_ratio: Optional[float] = None
    valid_sampler_ratio: Optional[float] = None
    test_sampler_ratio: Optional[float] = None

    # Default No Class Weight
    class_weight: Optional[Tuple[float]] = None
    class_weight_transformer: Optional[Callable] = None

    # Default Don't Select Model
    model: Optional[Type[torch.nn.Module]] = None
    model_args: Dict[str, Any] = field(default_factory=dfac_model_args)

    # Default No Encoder
    encoder_root: Optional[str] = None
    encoder: Optional[Type[torch.nn.Module]] = None

    # The classifiers from supconresnet.py
    supcon_group_list: Optional[Tuple[str]] = None

    # Default model save root
    checkpoint_root: Optional[str] = None

    # Default Select Adam as Optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam  # type: ignore
    optimizer_args: Dict[str, Any] = field(default_factory=dfac_dataset_optimizer_args)

    # Default adjust learning rate
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None  # pylint: disable=protected-access
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Set number of epochs to train
    num_epochs: int = 5

    # Set model save threshold
    model_save_threshold: float = 1.0

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)

    # For sweep. Modify the key if sweep config changes (sweep yaml is important)
    def update_value(self, key, value):
        if key == "lr":
            self.optimizer_args["lr"] = value
        elif key == "model":
            # design for class method initialization design model
            model_parms = value.split('.')
            _model = models
            for p in model_parms:
                assert hasattr(_model, p)
                _model = getattr(_model, p)
            self.model = _model
        elif key == "model_name":
            self.model_args["model_name"] = value
        elif key == "optimizer":
            assert hasattr(torch.optim, value)
            self.optimizer = getattr(torch.optim, value)
        elif key == "lr_scheduler":
            assert hasattr(torch.optim.lr_scheduler, value)
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, value)
        elif key == "gamma":
            self.lr_scheduler_params[key] = value
        elif hasattr(self, key):
            assert getattr(self, key) is None or isinstance(getattr(self, key), type(value)), (key, value)
            setattr(self, key, value)
        else:
            raise NotImplementedError(f"Unknown key={key}")

    def update_sweep_dict(self, wandb_config: Dict[str, Any]):
        SWEEP_ARG_PREFIX = "WS_"
        for k, v in wandb_config.items():
            if k.startswith(SWEEP_ARG_PREFIX):
                sp_name = k[len(SWEEP_ARG_PREFIX):]
                # wandb.config.as_dict() returns Dict[k, Dict[str, v]]
                # https://github.com/wandb/client/blob/master/wandb/wandb_config.py#L321
                sp_value = v['value']
                self.update_value(sp_name, sp_value)
