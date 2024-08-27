# LightDPH

This repository contains the source code for **LightDPH**, a project designed for encoder training and classification tasks, particularly in the context of dermatoscopic images and other medical imaging data. The project is divided into two main components: the `encoder` module and the `classification` module.

## Project Structure

### Encoder Module
The `encoder` module is responsible for training encoders that can be used for feature extraction in downstream tasks. This module includes the following key components:

- **Datasets**: Handles the loading and preprocessing of datasets. It includes utilities for transforming images and managing different dataset formats.
- **Networks**: Contains various network architectures, such as ResNet and EfficientNet, that are used for encoding.
- **Training**: Implements the training loop and loss functions necessary for encoder training.
- **Utilities**: Provides helper functions for tasks like learning rate adjustment, saving models, and computing metrics.

Key script:
- `run_encoder_training.py`: The main script to run the encoder training process.

### Classification Module
The `classification` module is designed for training and evaluating classification models. This is particularly useful for medical image classification tasks.

- **Evaluate**: Includes functions to evaluate model performance using various metrics.
- **Experiment**: Contains configurations and functions for setting up and running experiments.
- **Models**: Provides implementations of different classification models, such as ResNet, EfficientNet, and custom architectures like LHAC.
- **Training**: Contains the training loop and loss functions specific to classification tasks.
- **Preprocessing**: Handles image preprocessing steps that are necessary before feeding data into models.

Key script:
- `run_experiment.py`: The main script to run classification experiments.


## Credits

This repository builds upon [SupContrast: Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast).

Huge thanks to the original developers for providing a solid foundation to work from. We highly recommend checking out the original repository for more insights.
