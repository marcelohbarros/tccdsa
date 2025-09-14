import argparse
import os

import numpy as np
import pandas
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

import metrics as m
import csvwriter as cw
import config as cfg
from log import print_verbose


def load_data(data_path, metrics):
    def get_db_from_file(full_path):
        print_verbose(f"Reading data from: {full_path}")
        csv_file_metrics = m.FileMetrics(full_path)
        tags = csv_file_metrics.get_tags(argument_metrics=metrics, validate=False)

        df = pandas.read_csv(full_path, usecols=tags)    
        return df, csv_file_metrics, tags

    data = {}
    if os.path.isdir(data_path):
        print_verbose(f"{data_path} is a directory. Reading all CSV files inside...")
        dir = data_path
        files = [f for f in os.listdir(dir) if f.endswith('.csv')]
        for file in files:
            full_path = os.path.join(dir, file)
            df, csv_file_metrics, tags = get_db_from_file(full_path)
            data[file] = (df, csv_file_metrics, tags)
    else:
        file = os.path.basename(data_path)
        df, csv_file_metrics, tags = get_db_from_file(data_path)
        data[file] = (df, csv_file_metrics, tags)
    print_verbose(f"Found {len(data)} dataset(s).")
    return data


def print_data_stats(df, tags, bug_tag):
    data_size = len(df)
    bug_module_count = len(df[df[bug_tag] == True])
    print_verbose(f"metrics: {tags}")
    print_verbose(f"data count: {data_size}")
    print_verbose(f"modules with bugs: {bug_module_count} ({(bug_module_count / data_size) * 100:.2f}%)")


def balance_data(df, bug_tag, balance_ratio, random_seed):
    """
    Upsample bug>0 rows in df until the ratio of bug=True to total is > balance_ratio.
    Returns the entire DataFrame (not just the training set).
    """
    print_verbose(f"Balancing data to have at most {balance_ratio*100:.1f}% of bug>0 samples in the training set...")
    bug_true = df[df[bug_tag] > 0]
    n_total = len(df)
    n_true = len(bug_true)
    ratio = n_true / n_total if n_total > 0 else 0
    print_verbose(f"Initial data: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
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
        print_verbose(f"Data balanced: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
    else:
        print_verbose(f"Data already balanced")
    return df


def split_data(df, train_len):
    print_verbose(f"Splitting data into {train_len*100:.1f}% train and {(1-train_len)*100:.1f}% validation sets...")
    rng = np.random.default_rng()
    random_seed = rng.integers(0, 2**32 - 1)
    train_df = df.sample(frac=train_len, random_state=random_seed)
    validation_df = df.drop(train_df.index)
    print_verbose(f"train data count: {len(train_df)}")
    print_verbose(f"validation data count: {len(validation_df)}")
    return train_df, validation_df, random_seed


def train_model(train_df, random_seed, name_tag, bug_tag, use_boolean_model):
    print_verbose("Training model...")
    feature_columns = [col for col in train_df.columns if col not in [name_tag, bug_tag]]
    X_train = train_df[feature_columns]
    y_train = train_df[bug_tag].astype(bool) if use_boolean_model else train_df[bug_tag]
    clf = RandomForestClassifier(random_state=random_seed)
    clf.fit(X_train, y_train)
    return clf, feature_columns


def evaluate_model(test_name, run_number, clf, feature_columns, validation_df, bug_tag, dataset, writer):
    print_verbose("Evaluating model...")
    print_verbose("-----")
    print_verbose(f"Results for dataset '{dataset}':")
    X_val = validation_df[feature_columns]
    y_val = validation_df[bug_tag].astype(bool)
    y_pred = clf.predict(X_val).astype(bool)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0, average=None)
    rec = recall_score(y_val, y_pred, zero_division=0, average=None)
    f1 = f1_score(y_val, y_pred, zero_division=0, average=None)
    cm = confusion_matrix(y_val, y_pred)
    print_verbose(f"Validation accuracy: {acc:.4f}")
    print_verbose(f"Validation precision (False, True): {prec}")
    print_verbose(f"Validation recall (False, True): {rec}")
    print_verbose(f"Validation F1-score (False, True): {f1}")
    print_verbose("Confusion matrix:")
    print_verbose(cm)
    print_verbose("Classification report:")
    print_verbose(classification_report(y_val, y_pred, zero_division=0))

    # Calculate AUC if possible
    try:
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(validation_df[feature_columns])[:, 1]
            auc = roc_auc_score(validation_df[bug_tag].astype(bool), y_proba)
            print_verbose(f"Validation AUC: {auc:.4f}")
        else:
            print("AUC cannot be calculated: classifier does not support predict_proba.")
    except Exception as e:
        print(f"Error calculating AUC: {e}")

    save_results_to_csv(writer, test_name, run_number, dataset, acc, prec, rec, f1, cm, auc)


def normalize_data(df, name_tag, bug_tag):
    print_verbose("Normalizing data...")
    feature_columns = [col for col in df.columns if col not in [name_tag, bug_tag]]
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def extract_features(df, name_tag, bug_tag, features_number):
    print_verbose("Extracting features...")

    feature_columns = [col for col in df.columns if col not in [name_tag, bug_tag]]
    x_features = df[feature_columns]
    pca = PCA(n_components=min(len(feature_columns), features_number))
    principal_components = pca.fit_transform(x_features)
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    df_pc = pandas.DataFrame(data=principal_components, columns=pc_columns)
    df = pandas.concat([df[[name_tag, bug_tag]].reset_index(drop=True), df_pc.reset_index(drop=True)], axis=1)

    return df


def generate_random_seed():
    rng = np.random.default_rng()
    return rng.integers(0, 2**32 - 1)


def save_results_to_csv(writer, *row_data):
    print_verbose("Saving results to CSV...")
    row = cw.CsvRowData(*row_data)
    writer.write(row.to_dict())


def main():
    random_seed = generate_random_seed()
    writer = cw.CsvWriter('log/results.csv')

    current_test = 0

    for preset in cfg.presets:
        print("\r============================================")
        print(f"Using configuration: {preset.name} - {preset.description}")

        test_name = preset.name
        argument_metrics = m.ArgumentMetrics(preset.metrics, validate=True)
        train_len = preset.train_len
        balance_ratio = preset.balance_ratio
        use_boolean_model = preset.use_boolean_model
        features_number = preset.pca_features
    
        data = load_data(cfg.data_path, argument_metrics)
        number_of_tests = len(cfg.presets) * len(data)

        for dataset, (df, csv_file_metrics, tags) in data.items():
            current_test += 1
            print(f"\rRunning test {current_test} of {number_of_tests}...")

            for run_number in range(0, cfg.repetitions):
                print(f'\r[{"*" * run_number}{" " * (cfg.repetitions - run_number)}]', end='')
                print_verbose(f"Processing dataset: {dataset}")
                name_tag = csv_file_metrics.get_name_tag()
                bug_tag = csv_file_metrics.get_bug_tag()
                print_data_stats(df, tags, bug_tag)
                df = normalize_data(df, name_tag, bug_tag)
                if features_number:
                    df = extract_features(df, name_tag, bug_tag, features_number)
                train_df, validation_df, random_seed = split_data(df, train_len)
                train_df = balance_data(train_df, bug_tag, balance_ratio, random_seed)
                clf, feature_columns = train_model(train_df, random_seed, name_tag, bug_tag, use_boolean_model)
                evaluate_model(test_name, run_number, clf, feature_columns, validation_df, bug_tag, dataset, writer)
            print(f"\r{' ' * (cfg.repetitions + 2)}", end='')  # Clear progress bar line

if __name__ == "__main__":
    main()
