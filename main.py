import argparse
import pandas
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import metrics as m
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Read a CSV file and print its contents.")
    parser.add_argument('--data_path', type=str, default='tccdsa/datasets/tomcat.csv', help='Path to the CSV file to open (default: datasets/tomcat.csv)')
    parser.add_argument('--metrics', type=str, nargs='+', default=[], help='List of column names to read from the CSV file (default: all columns)')
    parser.add_argument('--train_len', type=float, default=0.7, help='Proportion of data to use for training (default: 0.7)')
    parser.add_argument('--balance_ratio', type=float, default=0.5, help='If set, upsample bug=True rows until the ratio is equal to the balance_ratio (default: 0.5)')
    parser.add_argument('--use_boolean_model', action='store_true', help='If set, use a model that predicts bugs based on booleans instead of ints (default: False)')

    args = parser.parse_args()
    metrics = m.ArgumentMetrics(args.metrics, validate=True)

    return args.data_path, metrics, args.train_len, args.balance_ratio, args.use_boolean_model


def load_data(data_path, metrics):
    print(f"Reading data from: {data_path}")
    csv_file_metrics = m.FileMetrics(data_path)
    tags = csv_file_metrics.get_tags(argument_metrics=metrics, validate=False)

    df = pandas.read_csv(data_path, usecols=tags)    

    return df, csv_file_metrics, tags


def print_data_stats(df, tags, bug_tag):
    data_size = len(df)
    bug_module_count = len(df[df[bug_tag] == True])
    print(f"metrics: {tags}")
    print(f"data count: {data_size}")
    print(f"modules with bugs: {bug_module_count} ({(bug_module_count / data_size) * 100:.2f}%)")


def balance_data(df, bug_tag, balance_ratio, random_seed):
    """
    Upsample bug>0 rows in df until the ratio of bug=True to total is > balance_ratio.
    Returns the entire DataFrame (not just the training set).
    """
    print(f"Balancing data to have at most {balance_ratio*100:.1f}% of bug>0 samples in the training set...")
    bug_true = df[df[bug_tag] > 0]
    n_total = len(df)
    n_true = len(bug_true)
    ratio = n_true / n_total if n_total > 0 else 0
    print(f"Initial data: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
    if n_true > 0 and ratio <= balance_ratio:
        while True:
            n_true = len(df[df[bug_tag] > 0])
            n_total = len(df)
            ratio = n_true / n_total
            if ratio > balance_ratio:
                break
            n_to_add = min(n_true, int(n_total * balance_ratio) - n_true)
            n_to_add = max(1, n_to_add)
            bug_true_upsampled = bug_true.sample(n=n_to_add, replace=True, random_state=random_seed)
            df = pandas.concat([df, bug_true_upsampled], ignore_index=True)
        print(f"Data balanced: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
    else:
        print(f"Data already balanced")
    return df


def split_data(df, train_len):
    print(f"Splitting data into {train_len*100:.1f}% train and {(1-train_len)*100:.1f}% validation sets...")
    rng = np.random.default_rng()
    random_seed = rng.integers(0, 2**32 - 1)
    train_df = df.sample(frac=train_len, random_state=random_seed)
    validation_df = df.drop(train_df.index)
    print(f"train data count: {len(train_df)}")
    print(f"validation data count: {len(validation_df)}")
    return train_df, validation_df, random_seed


def train_model(train_df, random_seed, name_tag, bug_tag, use_boolean_model):
    print("Training model...")
    feature_columns = [col for col in train_df.columns if col not in [name_tag, bug_tag]]
    X_train = train_df[feature_columns]
    y_train = train_df[bug_tag].astype(bool) if use_boolean_model else train_df[bug_tag]
    clf = RandomForestClassifier(random_state=random_seed)
    clf.fit(X_train, y_train)
    return clf, feature_columns


def evaluate_model(clf, feature_columns, validation_df, bug_tag):
    print("Evaluating model...")
    X_val = validation_df[feature_columns]
    y_val = validation_df[bug_tag].astype(bool)
    y_pred = clf.predict(X_val).astype(bool)
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


def normalize_data(df, name_tag, bug_tag):
    print("Normalizing data...")
    feature_columns = [col for col in df.columns if col not in [name_tag, bug_tag]]
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def generate_random_seed():
    rng = np.random.default_rng()
    return rng.integers(0, 2**32 - 1)


def main():
    data_path, argument_metrics, train_len, balance_ratio, use_boolean_model = parse_args()
    random_seed = generate_random_seed()
    df, csv_file_metrics, tags = load_data(data_path, argument_metrics)
    name_tag = csv_file_metrics.get_name_tag()
    bug_tag = csv_file_metrics.get_bug_tag()
    print_data_stats(df, tags, bug_tag)
    df = normalize_data(df, name_tag, bug_tag)
    train_df, validation_df, random_seed = split_data(df, train_len)
    train_df = balance_data(train_df, bug_tag, balance_ratio, random_seed)
    clf, feature_columns = train_model(train_df, random_seed, name_tag, bug_tag, use_boolean_model)
    evaluate_model(clf, feature_columns, validation_df, bug_tag)


if __name__ == "__main__":
    main()
