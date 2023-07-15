import numpy as np

class Config():
    """
    hyper-parameters general configuration file
    """
    # General hyper-parameters
    random_seed = 50
    lr = 5e-3
    n_epochs = 250
    momentum = 0.9
    betas = (0.9, 0.999)
    log_interval = 1
    weight_decay = 1e-2
    batch_size = 4
    num_workers = 1
    train_split = 0.8
    classification_thresholds = np.arange(0.1, 1, 0.1)
    single_prediction_threshold = 0.5
    warmup_epochs = 0
    step_size = 10
    gamma = 0.94
    epsilon = 1e-8

    # Logs directory path
    logs_path = r"..."
    # Data directory path
    csv_path = r"..."
    # Tensorboard log files path
    tensorboard_log_path = r"..."

    # Output data
    num_classes = 1

    # Logging
    log_clearML = False
    log_roc_epoch = 5

    # Models
    models = ['resnetv2_50']
    fc_in_features_num = 1000
