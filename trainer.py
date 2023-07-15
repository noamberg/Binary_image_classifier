from utils import CountMeter, accuracy_score, apply_thresholds, calculate_sensitivity, calculate_specificity, \
     log_batch_results, log_epoch_results, calc_epoch_metrics
import numpy as np
from torch import nn
import torch
import os
from config import Config
from sklearn.metrics import roc_curve, roc_auc_score, recall_score

args = Config()

def train(train_loader, model, criterion, optimizer, epoch, address, device):
    """
    train the network for selected epoch iteration
    @param data_loader: train dataloader
    @param model: selected model
    @param threshold: sigmoid threshold
    @param criterion: loss fucntion
    @param optimizer: optimizer type
    @param scheduler: learning rate scheduler
    @param epoch: epoch number
    @param addr: log folder path
    @param device: cuda device
    @param data_type: histogram / image
    @return: results dictionary
    """
    # Training mode (backpropagate and update weights)
    model.train()
    m = nn.Sigmoid()

    # Reset countmeters
    losses = CountMeter()
    accuracies = CountMeter()
    sensitivities = CountMeter()
    specificities = CountMeter()
    balanced_accuracies = CountMeter()
    predictions = CountMeter()
    ground_truths = CountMeter()
    probabilities = CountMeter()

    step_counter = 0

    # Iterate over batches
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Foward-pass
        data = data.float().to(device)
        targets = target.unsqueeze(dim=1).float().to(device)

        # Forward pass
        logit = model(data)
        loss = criterion(logit, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent
        optimizer.step()

        # Calculate probabilities and threshold them through sigmoid
        outputs = m(logit).cpu().detach().numpy()

        # Convert predictions to probabilities and threshold them.
        # y_preds_all_thresholds = apply_thresholds(probabilities=outputs, thresholds=args.classification_thresholds)

        # Threshold predictions to create binary predictions
        y_preds = np.array(
            apply_thresholds(probabilities=outputs, thresholds=[args.single_prediction_threshold])).flatten()
        y_trues = target.cpu().detach().numpy().astype(int)

        # Calculate batch accuracy, specificity, sensitivity and balanced accuracy
        batch_accuracy = accuracy_score(y_preds, y_trues)
        batch_sensitivity = calculate_sensitivity(y_preds, y_trues)
        # Calculate sensitivity by scikit-learn
        batch_sensitivity_scikit = recall_score(y_trues, y_preds)
        batch_specificity = calculate_specificity(y_preds, y_trues)
        batch_balanced_accuracy = batch_sensitivity + batch_specificity / 2

        # Accumulate metrics
        probabilities.update(outputs)
        predictions.update(y_preds)
        ground_truths.update(y_trues)
        losses.update(loss.item())
        accuracies.update(batch_accuracy)
        sensitivities.update(batch_sensitivity)
        specificities.update(batch_specificity)
        balanced_accuracies.update(batch_balanced_accuracy)

        # Log results into batch logs file
        batches_logs_path = os.path.join(address, 'batches_logs')
        if not os.path.exists(os.path.join(address, 'batches_logs')):
            os.makedirs(batches_logs_path)

        batch_results = {'epoch': epoch, 'batch': batch_idx, 'loss': loss.item(), 'accuracy': batch_accuracy,
                         'sensitivity': batch_sensitivity, 'specificity': batch_specificity,
                         'balanced_accuracy': batch_balanced_accuracy}
        log_batch_results(address=batches_logs_path, results=batch_results, mode='train')

        # Print results if batch_idx is a multiple of log_interval
        if batch_idx % args.log_interval == 0:
            print(
                f"Mode: Train, Epoch [{epoch}/{args.n_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Batch Loss: {loss:.4f}, Batch accuracy: {batch_accuracy:.4f}, "
                f"Batch sensitivity: {batch_sensitivity:.4f}, Batch specificity: {batch_specificity:.4f}, Batch balanced accuracy: {batch_balanced_accuracy:.4f}")

        # Save final embedding layer before classification head
        step_counter += 1

    # Calculate overall metrics for the epoch
    predictions_arr = [item for sublist in predictions.value() for item in
                       sublist]  # np.array(predictions.value()).flatten()
    ground_truths_arr = [item for sublist in ground_truths.value() for item in sublist]
    probabilities_arr = [item for sublist in probabilities.value() for item in sublist]

    epoch_accuracy, epoch_sensitivity, epoch_specificity, epoch_balanced_accuracy = \
        calc_epoch_metrics(predictions_arr, ground_truths_arr)

    # Log results into epoch logs file
    epochs_logs_path = os.path.join(address, 'epochs_logs')
    if not os.path.exists(epochs_logs_path):
        os.makedirs(epochs_logs_path)

    # Log epoch results
    log_epoch_results(address=epochs_logs_path,
                      results=dict(epoch=epoch, loss=losses.avg(), accuracy=epoch_accuracy,
                                   sensitivity=epoch_sensitivity,
                                   specificity=epoch_specificity,
                                   balanced_accuracy=epoch_balanced_accuracy), mode='train')

    train_results_dictionary = {'epoch': epoch, 'loss': losses.avg(), 'accuracy': epoch_accuracy, 'sensitivity': epoch_sensitivity, 'specificity': epoch_specificity, 'balanced_accuracy': epoch_balanced_accuracy}
    # Return epoch metrics
    return train_results_dictionary


def validate(val_loader, model, criterion, epoch, address, device):
    """
    train the network for selected epoch iteration
    @param data_loader: train dataloader
    @param model: selected model
    @param threshold: sigmoid threshold
    @param criterion: loss fucntion
    @param optimizer: optimizer type
    @param scheduler: learning rate scheduler
    @param epoch: epoch number
    @param addr: log folder path
    @param device: cuda device
    @param data_type: histogram / image
    @return: results dictionary
    """
    model.eval()
    m = nn.Sigmoid()

    # Reset countmeters
    losses = CountMeter()
    accuracies = CountMeter()
    sensitivities = CountMeter()
    specificities = CountMeter()
    balanced_accuracies = CountMeter()
    predictions = CountMeter()
    ground_truths = CountMeter()
    probabilities = CountMeter()

    step_counter = 0

    with torch.no_grad():
        # Iterate over batches
        for batch_idx, (data, target) in enumerate(val_loader):
            # Foward-pass
            data = data.float().to(device)
            targets = target.unsqueeze(dim=1).float().to(device)

            # Forward pass
            logit = model(data)
            loss = criterion(logit, targets)

            # Calculate probabilities and threshold them through sigmoid
            outputs = m(logit).cpu().detach().numpy()

            # Convert predictions to probabilities and threshold them.
            # y_preds_all_thresholds = apply_thresholds(probabilities=outputs, thresholds=args.classification_thresholds)

            # Threshold predictions to create binary predictions
            y_preds = np.array(
                apply_thresholds(probabilities=outputs, thresholds=[args.single_prediction_threshold])).flatten()
            y_trues = target.cpu().detach().numpy().astype(int)

            # Calculate batch accuracy, specificity, sensitivity and balanced accuracy
            batch_accuracy = accuracy_score(y_preds, y_trues)
            batch_sensitivity = calculate_sensitivity(y_preds, y_trues)
            # Calculate sensitivity by scikit-learn
            batch_specificity = calculate_specificity(y_preds, y_trues)
            batch_balanced_accuracy = batch_sensitivity + batch_specificity / 2

            # Accumulate metrics
            probabilities.update(outputs)
            predictions.update(y_preds)
            ground_truths.update(y_trues)
            losses.update(loss.item())
            accuracies.update(batch_accuracy)
            sensitivities.update(batch_sensitivity)
            specificities.update(batch_specificity)
            balanced_accuracies.update(batch_balanced_accuracy)

            # Log results into batch logs file
            batches_logs_path = os.path.join(address, 'batches_logs')
            if not os.path.exists(batches_logs_path):
                os.makedirs(batches_logs_path)

            batch_results = {'epoch': epoch, 'batch': batch_idx, 'loss': loss.item(), 'accuracy': batch_accuracy,
                             'sensitivity': batch_sensitivity, 'specificity': batch_specificity,
                             'balanced_accuracy': batch_balanced_accuracy}
            log_batch_results(address=batches_logs_path, results=batch_results, mode='validation')

            # Print results if batch_idx is a multiple of log_interval
            if batch_idx % args.log_interval == 0:
                print(
                    f"Mode: Validation, Epoch [{epoch}/{args.n_epochs}], Step [{batch_idx+1}/{len(val_loader)}], Batch Loss: {loss:.4f}, Batch accuracy: {batch_accuracy:.4f}, "
                    f"Batch sensitivity: {batch_sensitivity:.4f}, Batch specificity: {batch_specificity:.4f}, Batch balanced accuracy: {batch_balanced_accuracy:.4f}")

            # Save final embedding layer before classification head
            step_counter += 1

    # Calculate overall metrics for the epoch
    predictions_arr = [item for sublist in predictions.value() for item in
                       sublist]  # np.array(predictions.value()).flatten()
    ground_truths_arr = [item for sublist in ground_truths.value() for item in sublist]
    probabilities_arr = [item for sublist in probabilities.value() for item in sublist]

    epoch_accuracy, epoch_sensitivity, epoch_specificity, epoch_balanced_accuracy = \
        calc_epoch_metrics(predictions_arr, ground_truths_arr)

    # Log results into epoch logs file
    epochs_logs_path = os.path.join(address, 'epochs_logs')
    if not os.path.exists(epochs_logs_path):
        os.makedirs(epochs_logs_path)

    # Log epoch results
    log_epoch_results(address=epochs_logs_path,
                      results=dict(epoch=epoch, loss=losses.avg(), accuracy=epoch_accuracy,
                                   sensitivity=epoch_sensitivity,
                                   specificity=epoch_specificity,
                                   balanced_accuracy=epoch_balanced_accuracy), mode='validation')

    # Create and log ROC curve
    if epoch % args.log_roc_epoch == 0:
        # Create and log ROC curve
        fpr, tpr, thresholds = roc_curve(y_true=ground_truths_arr, y_score=probabilities_arr)
        tnr = 1 - fpr
        auc = roc_auc_score(y_true=ground_truths_arr, y_score=probabilities_arr)

        import matplotlib.pyplot as plt
        # Plot ROC curve and Specificity-Sensitivity curve
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        ax[0].plot(fpr, tpr)
        ax[0].plot([0, 1], [0, 1], linestyle='--')
        ax[0].set_xlabel('FPR (1 - Specificity)')
        ax[0].set_ylabel('TPR (Sensitivity)')
        ax[0].set_title('ROC Curve')
        ax[0].legend(['AUC = %0.4f' % auc])
        ax[1].plot(tnr, tpr)
        ax[1].set_xlabel('TNR (Specificity)')
        ax[1].set_ylabel('TPR (Sensitivity)')
        ax[1].set_title('Specificity-Sensitivity Curve')
        ax[1].legend(['AUC = %0.4f' % auc])
        plt.tight_layout()
        plt.savefig(os.path.join(address, f'ROC_{epoch}.png'))

    validation_results_dict = {'epoch': epoch, 'loss': losses.avg(), 'accuracy': epoch_accuracy, 'sensitivity': epoch_sensitivity, 'specificity': epoch_specificity, 'balanced_accuracy': epoch_balanced_accuracy}

    # Return epoch metrics
    return validation_results_dict


