from utils import generate_seed, mkdir_if_missing, log_meters_to_ClearML, log_meters_to_writer, count_parameters, count_parameters_per_layer
from splitter import split_train_val
from dataset import Train, Validation, get_balanced_dataloader
from torch.utils.data import DataLoader
import torch
import datetime
import torch.nn as nn
from clearml import Task, Logger
import os
from config import Config
from trainer import train, validate

def main(args):
    # Generate seeds for reproducibility
    generate_seed(args.random_seed)

    # Set the logger
    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    print("date and time =", dt_string)

    # Set run name
    run_id = str(f'{args.models[0]}_{args.data_types[0]}_{dt_string}')
    print(run_id)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    assert device.type == 'cuda', 'GPU is not available'

    # Create log folder with current time and date
    addr = os.path.join(args.ds_logs_path, dt_string)

    # mkdir if not exists
    mkdir_if_missing(addr)

    # Create train and validation dataset folders
    train_val_save_path = os.path.join(addr, 'train_val_data')
    mkdir_if_missing(train_val_save_path)

    # Split the data to train and validation
    train_df, val_df = split_train_val(args.csv_path, args.train_split)
    train_csv_path = os.path.join(train_val_save_path, 'train.csv')
    val_csv_path = os.path.join(train_val_save_path, 'val.csv')
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    # Creating PT data samplers and loaders:
    augmentation_save_path = os.path.join(addr, 'augmentation_data')
    train_dataset = Train(csv_path=train_csv_path, aug_save_path=augmentation_save_path)
    val_dataset = Validation(csv_path=val_csv_path)

    training_dataloader = get_balanced_dataloader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    validation_dataloader = get_balanced_dataloader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Create the model
    import timm
    network = timm.create_model(args.models[0], pretrained=True).to(device)
    net_trainable_params = count_parameters(network)
    print(f"Total Network's Parameters Number is: {net_trainable_params}")
    print(count_parameters_per_layer(network))
    # Serialize final layer
    network.fc = nn.Linear(in_features=args.fc_in_features_num, out_features=1, bias=True).to(device)
    network = nn.Sequential(network, network.fc).to(device)

    # Set an optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas,
                                 eps=args.epsilon)

    # Set a loss function
    criterion = nn.BCEWithLogitsLoss()

    # Save the hyperparameters config file of this run to the log folder
    with open(os.path.join(addr, f'{dt_string}_config_file.txt'), 'w') as f:
        # Write the config args to the file
        var_names = [var_name for var_name in dir(args) if
                     not callable(getattr(args, var_name)) and not var_name.startswith("__")]
        # write all var_names and their values
        for var_name in var_names:
            f.write(f'{var_name}: {getattr(args, var_name)}\n')

    # Set tensorboard writer and set its run name with run_id
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_log_path, run_id), flush_secs=10)

    # Train the model
    best_validation_accuracy = 0.0
    best_epoch = 0

    # Run loop for epochs
    for epoch in range(args.n_epochs):
        # Train the model
        print('\n-Train-\n')
        train_results_dictionary = train(train_loader=training_dataloader, model=network, criterion=criterion,
                                         optimizer=optimizer,
                                         epoch=epoch, address=addr, device=device)

        # Validate the model
        print('\n-Validation-\n')
        validation_results_dictionary = validate(val_loader=validation_dataloader, model=network, criterion=criterion,
                                                 epoch=epoch, address=addr, device=device)

        # Log the results
        log_meters_to_writer(train_results_dict=train_results_dictionary,
                             validation_results_dict=validation_results_dictionary, epoch=epoch, summary_writer=writer,
                             dt_time=dt_string)
        if args.log_clearML:
            log_meters_to_ClearML(train_results_dict=train_results_dictionary,
                                  validation_results_dict=validation_results_dictionary, epoch=epoch, logger=Logger)

        # Save the best model if validation accuracy is better than previous best
        if validation_results_dictionary['accuracy'] > best_validation_accuracy:
            best_validation_accuracy = validation_results_dictionary['accuracy']
            best_epoch = epoch

            # Save state dictionary, optimizer and scheduler states
            models_path = os.path.join(addr, 'models')
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            torch.save(network.state_dict(), os.path.join(models_path, f'best_{dt_string}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(models_path, f'best_optimizer.pth'))
            # torch.save(scheduler.state_dict(), os.path.join(models_path, f'best_scheduler.pth'))
            print(f'Epoch {str(best_epoch)} with accuracy {str(best_validation_accuracy)} has been saved.')

            # log saved epochs into txt file
            with open(os.path.join(models_path, f'{dt_string}_best_epoch.txt'), 'w') as f:
                f.write(f'Epoch {str(best_epoch)} with accuracy {str(best_validation_accuracy)} has been saved.')
                f.close()

    print(f'Finished to train {dt_string}\n')


if __name__ == '__main__':
    args = Config()
    main(args)
    print('Finished all training sessions.')
