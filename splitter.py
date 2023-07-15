import pandas as pd
import datetime
SEED = 42

# Create fucntion which splits dataset in to train and validation sets
def split_train_val(df_path, train_split):
    df = pd.read_csv(df_path)
    # Split the dataset into train and validation sets
    train_df = pd.DataFrame(columns=['file_name', 'label', 'min', 'max', 'mean', 'std'])
    val_df = pd.DataFrame(columns=['file_name', 'label', 'min', 'max', 'mean', 'std'])

    # Split the dataset into train and validation sets with each class having the same amount of samples
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_train_df = label_df.sample(frac=train_split, random_state=SEED)
        label_val_df = label_df.drop(label_train_df.index)

        train_df = train_df.append(label_train_df, ignore_index=True)
        val_df = val_df.append(label_val_df, ignore_index=True)

    return train_df, val_df


if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    print("date and time =", dt_string)

    train_split = 0.9
