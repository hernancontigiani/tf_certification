import os
import tensorflow as tf
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import datetime

from sklearn.metrics import classification_report, f1_score, accuracy_score

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def random_choice_image(folderpath):
    filename = random.choice(os.listdir(folderpath))
    filename = os.path.join(folderpath, filename)
    img = tf.io.read_file(filename)
    # Decode it into a tensor of 3 channels
    img = tf.image.decode_image(img, channels=3)
    return img, filename


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor of 3 channels
    img = tf.image.decode_image(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
    # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img

def pred_image(model, filename, class_names, plot=False, scale=True):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename, scale=scale)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

    msg = f"Prediction: {pred_class}"
    if plot == True:
        # Plot the image and predicted class
        plt.imshow(img)
        plt.title(msg)
        plt.axis(False);
    else:
        print(msg)

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.
    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"
    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

def create_modelcheckpoint_callback(experiment_name, dir_name="model_checkpoints", monitor="val_loss"):
    checkpoint_path = os.path.join(dir_name, f"{experiment_name}.ckpt")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    save_best_only=True,
                                    monitor=monitor # val_accuracy or val_loss
                                    #mode='max' # max for val_accuracy, min for val_loss
                                    )
    return checkpoint_callback

def history_plot(history, metric="accuracy"):
    epoch_count = range(1, len(history.history[metric]) + 1)
    plt.plot(epoch_count,  history.history[metric], label='train')
    val_metric = 'val_' + metric
    if val_metric in history.history:
        plt.plot(epoch_count,  history.history[val_metric], label='val')
    plt.legend()
    plt.show()

def history_fine_tune_plot(history, history_ft, metric="accuracy"):
    val_metric = 'val_' + metric
    initial_epochs = len(history.history[metric])
    acc = history.history[metric] + history_ft.history[metric]
    val = history.history[val_metric] + history_ft.history[val_metric]

    epoch_count = range(1, len(acc) + 1)
    plt.plot(epoch_count,  acc, label='train')
    plt.plot(epoch_count,  val, label='val')
    plt.plot([initial_epochs, initial_epochs],
                plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend()
    plt.show()

def unbatch_test_dataset(test_data):
    y_test = []
    for images, labels in test_ds.unbatch():
        if len(labels.shape) > 0:
	    # multicategorical
            val = labels.numpy().argmax()
        else:
	    # binary category
            val = labels.numpy()
        y_test.append(val)
    return y_test

def evaluate_classification(y_true, y_pred):    
    acc = accuracy_score(y_true, y_pred)
    if max(y_true) > 1: # multi categorical
        f1 = f1_score(y_true, y_pred, average="weighted")
    else:
        f1 = f1_score(y_true, y_pred)

    return {"accuracy": acc, "f1_score": f1 }

def muticlass_f1_score(y_true, y_pred, class_names):
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True)
    class_f1_scores = {}

    # Loop through classification report items
    for k, v in classification_report_dict.items():
        if k == "accuracy": # stop once we get to accuracy key
            break
        else:
            # Append class names and f1-scores to new dictionary
            class_f1_scores[class_names[int(k)]] = v["f1-score"]
    
    f1_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                          "f1_score": list(class_f1_scores.values())}).sort_values("f1_score", ascending=False)
    return f1_scores

def plot_f1_score(y_true, y_pred, class_names):
    f1_scores = muticlass_f1_score(y_true, y_pred, class_names)
    fig, ax = plt.subplots(figsize=(12, 25))
    scores = ax.barh(range(len(f1_scores)), f1_scores["f1_score"].values)
    ax.set_yticks(range(len(f1_scores)))
    ax.set_yticklabels(list(f1_scores["class_name"]))
    ax.set_xlabel("f1_score")
    ax.set_title("F1-Scores")
    ax.invert_yaxis(); # reverse the order

    def autolabel(rects): # Modified version of: https://matplotlib.org/examples/api/barchart_demo.html
        """
        Attach a text label above each bar displaying its height (it's value).
        """
        for rect in rects:
            width = rect.get_width()
            ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5, f"{width:.2f}",ha='center', va='bottom')

    autolabel(scores)
    plt.show()

def get_worse_pred(test_data, y_true, y_pred, y_prob, class_names, test_folder_path=None):
    filepaths =  []
    if test_folder_path is None:
        filepaths = test_data.file_paths
    else:
        for filepath in test_data.list_files(f"{test_folder_path}/*/*.jpg", shuffle=False):
            filepaths.append(filepath.numpy())
    
    pred_df = pd.DataFrame({"img_path": filepaths,
                           "y_true": y_true,
                           "y_pred": y_pred,
                           "pred_conf": y_prob.max(axis=1),
                           "y_true_classname": [class_names[i] for i in y_true],
                           "y_pred_classname": [class_names[i] for i in y_pred],
                            }
                           )

    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    # Get top 100 worse predictions
    top_wrong_pred_df = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:100]
    return top_wrong_pred_df


def plot_worse_pred(top_wrong_pred_df, start_index=0):
    images_to_view = 9
    plt.figure(figsize=(16,9))

    for i, row in enumerate(top_wrong_pred_df[start_index:(start_index+images_to_view)].itertuples()):
        plt.subplot(3,3, i+1)
        _, image_path, _, _, pred_conf, y_true_classname, y_pred_classname, _ = row
        img = load_and_prep_image(image_path)
        plt.imshow(img)
        plt.title(f"actual: {y_true_classname}, pred: {y_pred_classname} \nwith prob {pred_conf:.3f}")

    plt.show()

# Usa Mean absule error (MAE) scaled (MASE)
# MASE implemented courtesy of sktime - https://github.com/alan-turing-institute/sktime/blob/ee7a06843a44f4aaec7582d847e36073a9ab0566/sktime/performance_metrics/forecasting/_functions.py#L16
def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Find MAE of naive forecast (no seasonality)
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shifting of 1 day)

    return mae / mae_naive_no_season

def forescast_evaluate_preds(y_true, y_pred):
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    # Account for different sized metrics (for longer horizons, reduce to single number)
    if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}
