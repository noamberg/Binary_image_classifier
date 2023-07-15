import csv
import numpy as np
import os
import errno
import torch
import random
from config import Config
args = Config()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model):
  """Counts the number of parameters in each layer of a model.

  Args:
    model: The model to count the parameters of.

  Returns:
    A dictionary mapping layer names to the number of parameters in each layer.
  """
  layer_params = {}
  for name, param in model.named_parameters():
    if param.requires_grad:
      layer_name = name.split('.')[0]
      if layer_name not in layer_params:
        layer_params[layer_name] = 0
      layer_params[layer_name] += param.numel()

  return layer_params


def calc_epoch_metrics(predictions_arr, ground_truths_arr):
    """
    Calculate the metrics for the epoch
    :param predictions_arr: all the predictions for the epoch
    :param ground_truths_arr:  all the ground truths for the epoch
    :return: calculated metrics for the epoch
    """
    TN, FP, FN, TP = calc_metrics(ground_truths_arr, predictions_arr, save_path=None)

    epoch_accuracy = (TP + TN) / (TP + TN + FP + FN)
    epoch_sensitivity = TP / (TP + FN)
    epoch_specificity = TN / (TN + FP)
    epoch_balanced_accuracy = (epoch_sensitivity + epoch_specificity) / 2

    return epoch_accuracy, epoch_sensitivity, epoch_specificity, epoch_balanced_accuracy


def log_meters_to_ClearML(train_results_dict, validation_results_dict, epoch, logger):
    # log metrics to ClearML if needed
    logger.current_logger().report_scalar(title=f'BCE Loss', series=f'Train Loss', value=train_results_dict['loss'],
                                          iteration=epoch)
    logger.current_logger().report_scalar(title=f'BCE Loss', series=f'Validation Loss',
                                          value=validation_results_dict['loss'], iteration=epoch)

    logger.current_logger().report_scalar(title=f'Accuracy', series=f'Train Accuracy',
                                          value=train_results_dict['accuracy'], iteration=epoch)
    logger.current_logger().report_scalar(title=f'Accuracy', series=f'Validation Accuracy',
                                          value=validation_results_dict['accuracy'], iteration=epoch)

    logger.current_logger().report_scalar(title=f'Sensitivity', series=f'Train Sensitivity',
                                          value=train_results_dict['sensitivity'], iteration=epoch)
    logger.current_logger().report_scalar(title=f'Sensitivity', series=f'Validation Sensitivity',
                                          value=validation_results_dict['sensitivity'], iteration=epoch)

    logger.current_logger().report_scalar(title=f'Specificity', series=f'Train Specificity',
                                          value=train_results_dict['specificity'], iteration=epoch)
    logger.current_logger().report_scalar(title=f'Specificity', series=f'Validation Specificity',
                                          value=validation_results_dict['specificity'], iteration=epoch)

    logger.current_logger().report_scalar(title=f'Balanced Accuracy', series=f'Train Balanced Accuracy',
                                          value=train_results_dict['balanced_accuracy'], iteration=epoch)
    logger.current_logger().report_scalar(title=f'Balanced Accuracy', series=f'Validation Balanced Accuracy',
                                          value=validation_results_dict['balanced_accuracy'], iteration=epoch)
    return logger


def log_meters_to_writer(train_results_dict, validation_results_dict, epoch, summary_writer, dt_time):
    # log metrics to tensorboard
    summary_writer.add_scalars('Loss',
                               {"train": train_results_dict['loss'], "validation": validation_results_dict['loss']},
                               epoch)
    summary_writer.add_scalars('Accuracy', {"train": train_results_dict['accuracy'],
                                            "validation": validation_results_dict['accuracy']}, epoch)
    summary_writer.add_scalars('Sensitivity', {"train": train_results_dict['sensitivity'],
                                               "validation": validation_results_dict['sensitivity']}, epoch)
    summary_writer.add_scalars('Specificity', {"train": train_results_dict['specificity'],
                                               "validation": validation_results_dict['specificity']}, epoch)
    summary_writer.add_scalars('Balanced Accuracy', {"train": train_results_dict['balanced_accuracy'],
                                                     "validation": validation_results_dict['balanced_accuracy']}, epoch)
    return True


def log_epoch_results(address, results, mode):
    # Log it into epoch logs csv file
    if mode == 'train':
        csv_address = os.path.join(address, 'train_epoch_logs.csv')
        if not os.path.exists(csv_address):
            column_names = ['epoch', 'loss', 'accuracy', 'sensitivity', 'specificity', 'balanced_accuracy']
            create_csv(csv_address, column_names)
        # append the results dictionary to the csv file
        append_dict_as_row(os.path.join(address, 'train_epoch_logs.csv'), results)
    elif mode == 'validation':
        csv_address = os.path.join(address, 'val_epoch_logs.csv')
        if not os.path.exists(csv_address):
            column_names = ['epoch', 'loss', 'accuracy', 'sensitivity', 'specificity', 'balanced_accuracy']
            create_csv(csv_address, column_names)
        # append the results dictionary to the csv file
        append_dict_as_row(os.path.join(address, 'val_epoch_logs.csv'), results)
    return True


def log_batch_results(address, results, mode):
    # Log it into batch logs csv file
    if mode == 'train':
        csv_address = os.path.join(address, 'train_batch_logs.csv')
        if not os.path.exists(os.path.join(address, 'train_batch_logs.csv')):
            column_names = ['epoch', 'batch', 'loss', 'batch_accuracy', 'sensitivity', 'specificity',
                            'balanced_accuracy']
            create_csv(csv_address, column_names)

        # append the results dictionary to the csv file
        append_dict_as_row(csv_address, results)
        # append_dictionary_to_csv(dictionary=results, csv_file_path=csv_address)
    elif mode == 'validation':
        csv_address = os.path.join(address, 'val_batch_logs.csv')
        if not os.path.exists(os.path.join(address, 'validation_batch_logs.csv')):
            column_names = ['epoch', 'batch', 'loss', 'batch_accuracy', 'sensitivity', 'specificity',
                            'balanced_accuracy']
            create_csv(csv_address, column_names)

        # append the results dictionary to the csv file
        append_dict_as_row(csv_address, results)
    return True


def append_dict_as_row(file_path, results_dict):
    fieldnames = results_dict.keys()
    # Append results dictionary to csv file
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results_dict)


def calculate_specificity(predictions, ground_truth):
    true_negatives = sum([1 for pred, true in zip(predictions, ground_truth) if pred == 0 and true == 0])
    actual_negatives = len(ground_truth) - sum(ground_truth)

    specificity = true_negatives / actual_negatives if actual_negatives != 0 else 0
    return specificity


def calculate_sensitivity(predictions, ground_truth):
    sensitivity = None
    try:
        true_positives = sum([1 for pred, true in zip(predictions, ground_truth) if pred == 1 and true == 1])
        actual_positives = sum(ground_truth)

        sensitivity = true_positives / actual_positives if actual_positives != 0 else 0
    except ValueError:
        print('Error in calculating sensitivity')
    return sensitivity


def apply_thresholds(probabilities, thresholds):
    predictions = [[] for _ in range(len(probabilities))]
    # for each prediction run over the thresholds and apply the thresholds
    for i, probability in enumerate(probabilities):
        for threshold in thresholds:
            prediction = 1 if probability >= threshold else 0
            predictions[i].append(prediction)

    return predictions


def mkdir_if_missing(directory):
    """
    Make a directory if it doesn't exist
    @param directory:
    @return:
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def generate_seed(num):
    """
    Generate seeds for reproducibility
    @param num:
    @return:
    """
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)


def accuracy_score(y_true, y_pred):
    '''
    Calculate the accuracy score, in use.
    @param y_true: true labels
    @param y_pred: predicted labels
    @return: accuracy score
    '''
    return (y_true == y_pred).sum() / len(y_true)


def save_confusion_matrix(cm, save_path):
    """
    Save confusion matrix plot
    @param cm: confusion matrix
    @param save_path: save path
    @return: True
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()
    return True


def calc_metrics(y_trues, y_preds, save_path=None):
    """
    Calculate confusion matrix and its metrics
    @param y_trues: true labels
    @param y_preds: predicted labels
    @param save_path: save path
    @return: TN, FP, FN, TP
    """
    # Calculate metrics
    from sklearn import metrics
    confusion_matrix = metrics.confusion_matrix(y_trues, y_preds)
    if save_path is not None:
        save_confusion_matrix(confusion_matrix, save_path)

    TN, FP, FN, TP = confusion_matrix.ravel()

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    # # F1 score
    # F1 = 2 * PPV * TPR / (PPV + TPR)
    return TN, FP, FN, TP


def create_csv(path, rows=None):
    # rows = ['study_id', 'imd_id', 'approved', 'label', 'gixam_image_path', 'local_img_path']
    with open(path, 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(rows)
        f.close()


def append_dict_csv(data, csv_path):
    with open(csv_path, 'a', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(data.values())
        f.close()


class CountMeter(object):
    """
    Accumulate epoch predictions or targets
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.prevs = []

    def update(self, val):
        if isinstance(val, np.ndarray):
            self.val = list(val)
        elif isinstance(val, float):
            self.val = val
        self.prevs.append(self.val)

    def avg(self):
        return sum(self.prevs) / len(self.prevs)

    def value(self):
        return self.prevs

    def __len__(self):
        return len(self.prevs)
