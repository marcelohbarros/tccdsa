import os
import traceback

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


class Runner():
    _random_seed = None
    _data = None
    _number_of_tests = None
    _writer = None

    def __init__(self):
        raise ValueError("Not instantiable")

    @classmethod
    def random_seed(cls):
        if cls._random_seed is None:
            rng = np.random.default_rng()
            cls._random_seed = rng.integers(0, 2**32 - 1)
        return cls._random_seed

    @classmethod
    def data(cls):
        if cls._data is None:
            cls._data = cls.__load_data()
        return cls._data

    @classmethod
    def number_of_tests(cls):
        if cls._number_of_tests is None:
            cls._number_of_tests = len(cfg.presets) * len(cls.data())
        return cls._number_of_tests

    @classmethod
    def writer(cls):
        if cls._writer is None:
            cls._writer = cw.CsvWriter('log/results.csv')
        return cls._writer

    @classmethod
    def iter_tests(cls):
        for n, preset in enumerate(cfg.presets):
            test_id = n * 1000
            for dataset, (df, tags, name_tag, bug_tag) in cls.__get_filtered_data(preset).items():
                test_id += 1
                yield Test(test_id, dataset, df, tags, name_tag, bug_tag, preset)

    @classmethod
    def __get_filtered_data(cls, preset):
        filtered_data = {}
        for dataset, (df, csv_file_metrics) in cls.data().items():
            tags = csv_file_metrics.filter_tags(preset.metrics)
            name_tag = csv_file_metrics.get_name_tag()
            bug_tag = csv_file_metrics.get_bug_tag()
            if name_tag is None or bug_tag is None:
                raise ValueError(f"A name or bug column is missing in the dataset '{dataset}'.")
            if not tags:
                raise ValueError(f"No valid metrics found in the dataset '{dataset}' for the given preset metrics.")

            filtered_df = df[tags].copy()
            filtered_data[dataset] = (filtered_df, tags, name_tag, bug_tag)
        return filtered_data

    @classmethod
    def __load_data(cls):
        def get_db_from_file(full_path):
            print_verbose(f"Reading data from: {full_path}")
            csv_file_metrics = m.FileMetrics(full_path)

            df = pandas.read_csv(full_path)
            return df, csv_file_metrics

        data = {}
        data_path = cfg.data_path
        if os.path.isdir(data_path):
            print_verbose(f"{data_path} is a directory. Reading all CSV files inside...")
            dir = data_path
            files = [f for f in os.listdir(dir) if f.endswith('.csv')]
            for file in files:
                full_path = os.path.join(dir, file)
                df, csv_file_metrics = get_db_from_file(full_path)
                data[file] = (df, csv_file_metrics)
        else:
            file = os.path.basename(data_path)
            df, csv_file_metrics = get_db_from_file(data_path)
            data[file] = (df, csv_file_metrics)
        print_verbose(f"Found {len(data)} dataset(s).")
        return data


class Test():
    __count = 0

    def __init__(self, id, dataset, df, tags, name_tag, bug_tag, preset):
        self._id = id
        self._dataset = dataset
        self._df = df
        self._tags = tags
        self._name_tag = name_tag
        self._bug_tag = bug_tag
        self._preset = preset
        self._test_count = self.count()

    @classmethod
    def count(cls):
        cls.__count += 1
        return cls.__count

    def __iter__(self):
        print(f"\r({self.__count}/{Runner.number_of_tests()})")
        for run_number in range(0, cfg.repetitions):
            rep_id = 1000 * self._id + run_number
            #print(f'\r[{"*" * run_number}{" " * (cfg.repetitions - run_number)}]', end='')
            yield Model(rep_id, self._dataset, self._df, self._tags, self._name_tag, self._bug_tag, self._preset)
            #print(f"\r{' ' * (cfg.repetitions + 2)}", end='')  # Clear progress bar line


class Model():
    def __init__(self, rep_id, dataset, df, tags, name_tag, bug_tag, preset):
        self._rep_id = rep_id
        self._dataset = dataset
        self._df = df
        self._tags = tags
        self._name_tag = name_tag
        self._bug_tag = bug_tag
        self._test_id = rep_id // 1000
        self._run_number = rep_id % 1000
        self._preset = preset
        self._data = []

    def run(self):
        df = normalize_data(self._df, self._name_tag, self._bug_tag)
        df = extract_features(df, self._name_tag, self._bug_tag, self._preset.pca_features)
        train_df, validation_df, random_seed = split_data(df, self._preset.train_len)
        train_df = balance_data(train_df, self._bug_tag, self._preset.balance_ratio, random_seed)
        clf, feature_columns = train_model(train_df, random_seed, self._name_tag, self._bug_tag, self._preset.use_boolean_model)
        acc, prec, rec, f1, cm, auc = evaluate_model(clf, feature_columns, validation_df, self._bug_tag, self._dataset)
        
        self._data = [
            self._rep_id,
            self._test_id,
            self._run_number,
            self._preset.name,
            self._dataset,
            len(feature_columns),
            acc,
            prec,
            rec,
            f1,
            cm,
            auc
        ]

    def save_results(self):
        cw.save_results_to_csv(Runner.writer(), *self._data)


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


def evaluate_model(clf, feature_columns, validation_df, bug_tag, dataset):
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

    auc = None

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

    return acc, prec, rec, f1, cm, auc


def normalize_data(df, name_tag, bug_tag):
    print_verbose("Normalizing data...")
    feature_columns = [col for col in df.columns if col not in [name_tag, bug_tag]]
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def extract_features(df, name_tag, bug_tag, features_number):
    print_verbose("Extracting features...")
    if not features_number:
        return df

    feature_columns = [col for col in df.columns if col not in [name_tag, bug_tag]]
    x_features = df[feature_columns]
    pca = PCA(n_components=min(len(feature_columns), features_number))
    principal_components = pca.fit_transform(x_features)
    pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    df_pc = pandas.DataFrame(data=principal_components, columns=pc_columns)
    df = pandas.concat([df[[name_tag, bug_tag]].reset_index(drop=True), df_pc.reset_index(drop=True)], axis=1)

    return df


def main():
    try:
        for test in Runner.iter_tests():
            for model in test:
                model.run()
                model.save_results()

        print("\nAll tests completed.")

    # Ctrl C
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    #except Exception:
    #    print(f"\nAn error occurred: {traceback.format_exc()}")
    #    print(f"Ran {test_id} of {number_of_tests} tests.")
    finally:
        Runner.writer().close()

if __name__ == "__main__":
    main()
