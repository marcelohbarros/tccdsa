import argparse
import pandas
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Read a CSV file and print its contents.")
    parser.add_argument('--data_path', type=str, default='tccdsa/datasets/synapse-1.0.csv', help='Path to the CSV file to open (default: datasets/tomcat.csv)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['loc', 'avg_cc', 'cbo', 'rfc', 'wmc'], help='List of column names to read from the CSV file (default: all columns)')
    parser.add_argument('--train_len', type=float, default=0.7, help='Proportion of data to use for training (default: 0.7)')
    return parser.parse_args()


def load_data(data_path, metrics):
    required_columns = ['name', 'bug']
    print(f"Reading data from: {data_path}")
    if metrics:
        usecols = list(set(['name', 'bug'] + metrics))
        df = pandas.read_csv(data_path, usecols=usecols)
    else:
        df = pandas.read_csv(data_path)
        usecols = df.columns.to_list()
        metrics = set(usecols) - set(required_columns)
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"The following required columns are missing from the CSV file: {', '.join(missing_cols)}")
    
    df['bug'] = df['bug'].astype(bool)

    return df, metrics


def print_data_stats(df, metrics):
    data_size = len(df)
    bug_module_count = len(df[df['bug'] == True])
    print(f"metrics: {metrics}")
    print(f"data count: {data_size}")
    print(f"modules with bugs: {bug_module_count} ({(bug_module_count / data_size) * 100:.2f}%)")


def split_data(df, train_len):
    rng = np.random.default_rng()
    random_seed = rng.integers(0, 2**32 - 1)
    train_df = df.sample(frac=train_len, random_state=random_seed)
    validation_df = df.drop(train_df.index)

    # Keep duplicating bug=True points until true/false ratio is > 0.25
    balance_ratio = 0.5
    bug_true = train_df[train_df['bug'] == True]
    n_total = len(train_df)
    n_true = len(bug_true)
    ratio = n_true / n_total
    if n_true > 0 and ratio <= balance_ratio:
        while True:
            n_true = len(train_df[train_df['bug'] == True])
            n_total = len(train_df)
            ratio = n_true / n_total
            if ratio > balance_ratio:
                break
            n_to_add = min(n_true, int(n_total * balance_ratio) - n_true)
            n_to_add = max(1, n_to_add)
            bug_true_upsampled = bug_true.sample(n=n_to_add, replace=True, random_state=random_seed)
            train_df = pandas.concat([train_df, bug_true_upsampled], ignore_index=True)
        print(f"Training set balanced: {n_total} samples ({n_true} bug=True, {n_total - n_true} bug=False, ratio={ratio:.2f})")
    else:
        print(f"Training set already balanced or no bug=True samples to upsample.")
    print(f"train data count: {len(train_df)}")
    print(f"validation data count: {len(validation_df)}")
    return train_df, validation_df, random_seed


def train_model(train_df, random_seed):
    feature_columns = [col for col in train_df.columns if col not in ['name', 'bug']]
    X_train = train_df[feature_columns]
    y_train = train_df['bug']
    clf = RandomForestClassifier(random_state=random_seed)
    clf.fit(X_train, y_train)
    return clf, feature_columns


def evaluate_model(clf, feature_columns, validation_df):
    X_val = validation_df[feature_columns]
    y_val = validation_df['bug']
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0, average=None)
    rec = recall_score(y_val, y_pred, zero_division=0, average=None)
    f1 = f1_score(y_val, y_pred, zero_division=0, average=None)
    cm = confusion_matrix(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation precision (False, True): {prec}")
    print(f"Validation recall (False, True): {rec}")
    print(f"Validation F1-score (False, True): {f1}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(classification_report(y_val, y_pred, zero_division=0))


def main():
    args = parse_args()
    metrics = args.metrics if args.metrics else []
    data_path = args.data_path
    df, metrics = load_data(data_path, metrics)
    print_data_stats(df, metrics)
    train_df, validation_df, random_seed = split_data(df, args.train_len)
    clf, feature_columns = train_model(train_df, random_seed)
    evaluate_model(clf, feature_columns, validation_df)


if __name__ == "__main__":
    main()
