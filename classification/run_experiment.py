import pprint
import logging
import csv

from torchvision import transforms

import torch
import wandb

from DERMADL.utils import setup_logging
from DERMADL.experiment import ExperimentConfig, run_experiment
# from DERMADL.models import EfficientNet
# from DERMADL.models import resnet50
# from DERMADL.models import densenet121
# from DERMADL.models import se_resnext50_32x4d
# from DERMADL.models import SimSiam, SupConEfficientNet
# from DERMADL.models import arlnet50
# from DERMADL.training.loss_function import HACLoss
# from DERMADL.models import BCNN_Derm, effnetv2_s
from DERMADL.models import EHACv1, MLPClassifier, EfficientNet, SupConResNet, SupConEfficientNet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initiate Logger
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup Experiment Config
    config = ExperimentConfig()

    # Setup Data Source
    # config.data_path = '/mnt/data2/nfs/bennyhsu/isic2019_imgfolder/multi_train' # ISIC_2019 multiclass set
    # config.data_path = '/mnt/nfs_bdalab4_data/bennyhsu/mlhc/subset_multi' # ISIC_2019 multiclass subset
    # config.data_path = '/mnt/data2/nfs/bennyhsu/isic2019_imgfolder/subset' # ISIC_2019 binary training subset

    # config.train_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/subset_multi' # ISIC_2019 train subset
    # config.valid_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/subset_multi' # ISIC_2019 valid subset
    # config.test_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/subset_multi' # ISIC_2019 test subset

    # # pneumonia dataset
    # config.train_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/pneumonia/train'
    # config.valid_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/pneumonia/valid'
    # config.test_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/pneumonia/test'   

    # config.w_super = 0.05 #EHAC
    config.w_super = 0.0 # others
    config.exp_num = 'exp1'
    # skin dataset
    config.train_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin_3/five_fold/{config.exp_num}/train' # ISIC_2019 train
    config.valid_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin_3/five_fold/{config.exp_num}/valid' # ISIC_2019 valid
    config.test_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin_3/test' # ISIC_2019 test

    # skin pad-ufes
    # config.train_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin-pad-ufes/{config.exp_num}/train'
    # config.valid_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin-pad-ufes/{config.exp_num}/valid'
    # config.test_data_path = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin-pad-ufes/{config.exp_num}/valid'
    
    # save_folder = 'supcon'
    # save_folder = 'ehac'
    save_folder = 'hmce'
    backbone = 'effb1'
    backbone_full = 'efficientnet-b1'

    # Use specific loss function (default F.cross_entropy)
    # config.loss_function = HACLoss()
    
    # Setup GPU ID
    config.gpu_device_id = "0" # 0 is the first gpu of "visible ones"

    config.split_ratio = (0.8, 0.1, 0.1)

    config.num_epochs = 40
    config.batch_size = 16
    # config.class_weight = (43/119, 8/119, 37/119, 23/119, 18/119)

    # Set learning rate adjust
    config.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR  # type: ignore
    config.lr_scheduler_params = {
        'gamma': 0.95
    }

    # Setup Log Location
    config.tensorboard_log_root = "./tb/"
    config.wandb_dir = "./wandb/"

    # Setup Logging Group
    config.wandb_project = "ete_hc"
    # config.wandb_project = "multi-level-ete"
    # config.wandb_project = "pad-ufes"


    # config.wandb_group = "hac_loss"
    # config.wandb_group = "simclr"
    # config.wandb_group = "baseline"
    # config.wandb_group = "debug"
    # config.wandb_group = "sota"
    # config.wandb_group = "check_performance"
    # config.wandb_group = "0.1_lbda_weighted_loss"
    
    config.wandb_group = f'multi_level-{save_folder}-{config.exp_num}' # skin
    # config.wandb_group = 'baseline' # pna
    # config.wandb_group = 'ehac-exp1' # pna


    # Setup Random Seed
    config.random_seed = 828

    # Setup model save threshold
    config.model_save_threshold = 0.7

    # Setup W&B experiment name
    # wandb_exp_name = f"{config.cur_time}_projMLPWithHead_0.1superclass_hscloss_efficientnet-b4"
    # wandb_exp_name = f"{config.cur_time}_0.1superclass_hscloss_efficientnet-b4_888_continue200325"
    # wandb_exp_name = f"{config.cur_time}_supcon_mlpclassifier_efficientnet-b4_continue124324"
    # wandb_exp_name = f"{config.cur_time}_0.05superclass_hscloss_efficientnet-b4"
    # wandb_exp_name = f"prem_{w_super}_{config.cur_time}_efficientnet-b4_{config.exp_num}"
    # wandb_exp_name = f"{w_super}_{config.cur_time}_arlcnn50_{config.exp_num}"
    
    # wandb_exp_name = f'effnetB4-{config.cur_time}'
    wandb_exp_name = f'{config.w_super}{save_folder}-{config.cur_time}-{backbone}-ep110' # for new ehac

    # Decide Model
    # config.model = densenet121
    # config.model_args = {
    #     'pretrained': True
    # }

    # config.model = arlnet50
    # config.model_args = {
    #     "pretrained": True
    # }
    
    ## for skin
    config.model = MLPClassifier(name=backbone_full, n_class=8) # isic:8, ufes:6
    
    ## for pna
    # config.model = MLPClassifier(name='efficientnet-b4', n_class=4)


    # config.model = EfficientNet.from_pretrained  # from_pretrained_statedict
    # # config.model = EfficientNet.non_pretrained
    # config.model_args = {
    #     "model_name": "efficientnet-b4",
    #     # "checkpoint_path": "./models_save/20210306-010811/EfficientNet_ckpt_ep0099"
    # }

    # config.model = effnetv2_s

    # config.model = BCNN_Derm(m_class=2, n_class=8)

    # config.model = resnet50

    # config.model = se_resnext50_32x4d
    # config.model_args = {
    #     'pretrained': 'imagenet'
    # }

    config.supcon_group_list = ['HACEfficientNet', 'LinearClassifier', 
                                'MLPClassifier', 'HACDistanceClassifier',
                                'BCNN_Derm', 'BCNN_Origin']
    
    # HACLoss+CELoss
    # config.model = HACEfficientNet(name='efficientnet-b4', n_class=8, bsz=config.batch_size)
    

    # HAC+DistanceClassifier
    # bsz can be removed?
    # config.model = HACDistanceClassifier(name='efficientnet-b4', n_class=8, bsz=config.batch_size)
    # config.dist_alpha = 10



    # HSCLoss
    # config.encoder_root = '/home/bennyhsu/research/hmc/SupContrast-master/save/SupCon/path_models/0.05_HIL_path_efficientnet-b4_lr_0.001_bsz_32_seed_777/ckpt_epoch_60.pth'
    # config.encoder_root = f'/mnt/data2/nfs/bennyhsu/encoder_save/SupCon/path_models/prem_{w_super}_{config.exp_num}_HIL_path_efficientnet-b4_lr_0.001_bsz_32/last.pth'
    # config.encoder_root = '/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/path_models/0.05_exp3_HIL_path_efficientnet-b4_lr_0.001_bsz_16/ckpt_epoch_105.pth'
    
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-encoder_models/simsiam_path_efficientnet-b4_lr_0.001_bsz_16/last.pth'
    
    ## HAC, EHAC and SupCon
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-encoder_models/{config.w_super}_{config.exp_num}_EHAC_path_efficientnet-b4_lr_0.001_bsz_16/last.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}_models/{config.w_super}_{config.exp_num}_EHAC_path_resnet18_lr_0.001_bsz_16/last.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}_models/{config.w_super}_{config.exp_num}_EHAC_path_efficientnet-b1_lr_0.001_bsz_16/last.pth'
    ## multi-level
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-multi_level_models/{config.w_super}_{config.exp_num}_EHAC_path_{backbone_full}_lr_0.001_bsz_16/ckpt_epoch_60.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-multi_level_models/{config.w_super}_{config.exp_num}_EHAC_path_{backbone_full}_lr_0.001_bsz_16/last.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-multi_level-new_loss_models/{config.w_super}_{config.exp_num}_EHAC_path_efficientnet-b4_lr_0.001_bsz_16/ckpt_epoch_60.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-multi_level-new_loss_models/{config.w_super}_{config.exp_num}_EHAC_path_efficientnet-b4_lr_0.001_bsz_16/ckpt_epoch_15.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_EHAC_path_{backbone_full}_lr_0.001_bsz_16/last.pth'
    
    # pad-ufes dataset
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder_ufes/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_EHAC_path_{backbone_full}_lr_0.001_bsz_16/ckpt_epoch_90.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder_ufes/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_EHAC_path_{backbone_full}_lr_0.001_bsz_16/last.pth'
    
    ## SimSiam
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-encoder_models/{save_folder}_path_efficientnet-b4_lr_0.001_bsz_16_seed_777/last.pth'

    
    # SupCon
    # config.encoder_root = '/mnt/data2/nfs/bennyhsu/encoder_save/SupCon/path_models/SupCon_path_efficientnet-b4_lr_0.001_bsz_32_seed_666/last.pth'
    # config.encoder_root = f'/mnt/data2/nfs/bennyhsu/encoder_save/SupCon/path_models/{config.w_super}_{config.exp_num}_SupCon_path_efficientnet-b4_lr_0.001_bsz_32/last.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder_ufes/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_SupCon_path_{backbone_full}_lr_0.001_bsz_16/last.pth'
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_SupCon_path_{backbone_full}_lr_0.001_bsz_16/last.pth'

    # HMCE
    config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/model_save/encoder/{config.exp_num}-{save_folder}-{backbone}-encoder_models/{config.w_super}_{config.exp_num}_HMCE_path_{backbone_full}_lr_0.001_bsz_16/last.pth'

    
    
    # SimCLR
    # config.encoder_root = './DERMADL/encoder/SimCLR_path_resnet50_lr_0.05_decay_0.0001_bsz_12_temp_0.07_trial_0/ckpt_epoch_100.pth'
    # config.encoder_root = f'/mnt/data2/nfs/bennyhsu/encoder_save/SupCon/path_models/{w_super}_{config.exp_num}_SimCLR_path_efficientnet-b4_lr_0.001_bsz_32/last.pth'

    ## Pneumonia HAC, EHAC, and SupCon
    # config.encoder_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/pna_model_save/encoder/{config.exp_num}-{save_folder}-encoder-models/{config.w_super}_{config.exp_num}_EHAC_path_efficientnet-b4_lr_0.001_bsz_16/last.pth'



    if config.encoder_root is not None:
        if 'simsiam' in config.encoder_root:
            config.encoder = SimSiam(name=backbone_full)
        elif 'ehac' in config.encoder_root:
            config.encoder = EHACv1(name=backbone_full)
        else:
            config.encoder = SupConEfficientNet(name=backbone_full)
            # config.encoder = SupConResNet(name='resnet18')
    ## skin dataset transform
    config.train_transform = (
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.6678, 0.5300, 0.5245), (0.1320, 0.1462, 0.1573))
    )
    config.valid_transform = (
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.6678, 0.5300, 0.5245), (0.1320, 0.1462, 0.1573))
    )
    config.test_transform = (
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.6678, 0.5300, 0.5245), (0.1320, 0.1462, 0.1573))
    )

    ## pneumonia dataset transform
    # config.train_transform = (
    #     transforms.Resize([224, 224]),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ColorJitter(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # )
    # config.valid_transform = (
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # )
    # config.test_transform = (
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # )

    if 'pneumonia' in config.train_data_path:
        model_save = 'pna_model_save'
    elif 'ufes' in config.train_data_path:
        model_save = 'ufes_model_save'
    else:
        model_save = 'model_save'


    # Consider to use the following for tuning
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    # other learning rate adjust
    # config.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    # config.lr_scheduler_params = {
    #     "mode": 'min'
    # }

    # set optimizer
    config.optimizer_args = {
        "lr": 1e-3,
    }

    # Set model save path
    config.checkpoint_root = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/{model_save}/{save_folder}/{config.cur_time}'

    # Init logging
    setup_logging(f'./logs/{config.cur_time}.log', "DEBUG")
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=wandb_exp_name, group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict(),
    )
    wandb.tensorboard.patch(pytorch=True)

    # Run Experiment
    training_history, test_report, img_predict_results, best_model_id = run_experiment(config)

    # For the test set with labels
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # logger.info("Images Prediction Result:\n%s", pprint.pformat(img_predict_results))
    # img_truth_predict_csvpath = f"./raw_results_csv/analysis/{config.cur_time}_{best_model_id}.csv"
    img_truth_predict_csvpath = f"./raw_results_csv/testing_results/{config.cur_time}_{best_model_id}.csv"
    with open(img_truth_predict_csvpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # output prediction results with image name
        writer.writerow(['image_id', 'ground_truth', 'prediction'])
        ## using enumerate instead of range(len()) 
        for batch_idx in range(len(img_predict_results)):
            for i in range(config.batch_size):
                csv_row = []
                if i >= len(img_predict_results[batch_idx][0]):
                    break
                for j in range(3):
                    csv_row.append(img_predict_results[batch_idx][j][i])
                writer.writerow(csv_row)
