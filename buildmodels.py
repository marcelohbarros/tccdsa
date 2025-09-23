import os
import traceback

import pandas
import sklearn

import metrics as m
import csvhandler as ch
import config as cfg
from log import print_verbose, print_not_verbose


class Runner():
    _data = None
    _number_of_tests = None

    def __init__(self):
        raise ValueError("Not instantiable")

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
    _writer = None
    _count = 0

    def __init__(self, id, dataset, df, tags, name_tag, bug_tag, preset):
        self._id = id
        self._dataset = dataset
        self._df = df
        self._tags = tags
        self._name_tag = name_tag
        self._bug_tag = bug_tag
        self._preset = preset
        self._test_count = self.count()
        self._models = []

    @classmethod
    def count(cls):
        cls._count += 1
        return cls._count

    def __iter__(self):
        print(f"{'\r'*(not cfg.verbose)}Running test {self._count} of {Runner.number_of_tests()}... ({int(self._count / Runner.number_of_tests() * 100)}%)")
        for run_number in range(0, cfg.repetitions):
            rep_id = 1000 * self._id + run_number + 1
            print_not_verbose(f'\r[{"*" * run_number}{" " * (cfg.repetitions - run_number)}]', end='')
            model = Model(rep_id, self._df, self._tags, self._name_tag, self._bug_tag, self._preset)
            yield model
            self._models.append(model)
            print_not_verbose(f"\r{' ' * (cfg.repetitions + 2)}", end='')  # Clear progress bar line

    def save_results(self):
        if len(self._models) == 0:
            raise RuntimeError("No models were run for this test.")

        model_data = [model.data for model in self._models]

        data_list = [
            self._id,
            self._dataset,
            self._preset._name,
            self._preset.train_ratio,
            self._preset.balance_ratio,
            self._preset.use_boolean_model,
            self._preset.pca_features,
            len(self._df),
            model_data
        ]
        self.writer().write(data_list)

    @classmethod
    def writer(cls):
        if cls._writer is None:
            cls._writer = ch.TestCsvWriter()
        return cls._writer


class Model():
    _writer = None

    def __init__(self, rep_id, df, tags, name_tag, bug_tag, preset):
        self._rep_id = rep_id
        self._df = df
        self._tags = tags
        self._name_tag = name_tag
        self._bug_tag = bug_tag
        self._test_id = rep_id // 1000
        self._run_number = rep_id % 1000
        self._preset = preset
        self._data_list = []
        self._data_dict = {}

    def run(self):
        self._df = self.__normalize_data()
        self._df = self.__extract_features()
        train_df, validation_df = self.__split_data()
        train_df = self.__balance_data(train_df)
        clf, input_columns = self.__train_model(train_df)
        accuracy, precision, recall, f1, confusion_matrix, auc = self.__evaluate_model(clf, input_columns, validation_df)
        
        self._data_list = [
            self._rep_id,
            self._test_id,
            len(input_columns),
            accuracy,
            precision,
            recall,
            f1,
            confusion_matrix,
            auc
        ]

    def save_results(self):
        self._data_dict = self.writer().write(self._data_list)

    @classmethod
    def writer(cls):
        if cls._writer is None:
            cls._writer = ch.ModelCsvWriter()
        return cls._writer

    @property
    def data(self):
        return self._data_dict

    def __normalize_data(self):
        print_verbose("Normalizing data...")
        input_columns = [col for col in self._df.columns if col not in [self._name_tag, self._bug_tag]]
        scaler = sklearn.preprocessing.StandardScaler()
        df = self._df.copy()
        df[input_columns] = scaler.fit_transform(df[input_columns])
        return df

    def __extract_features(self):
        print_verbose("Extracting features...")
        if not self._preset.pca_features:
            return self._df

        feature_columns = [col for col in self._df.columns if col not in [self._name_tag, self._bug_tag]]
        x_features = self._df[feature_columns]
        pca = sklearn.decomposition.PCA(n_components=min(len(feature_columns), self._preset.pca_features))
        principal_components = pca.fit_transform(x_features)
        pc_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        df_pc = pandas.DataFrame(data=principal_components, columns=pc_columns)
        df = pandas.concat([self._df[[self._name_tag, self._bug_tag]].reset_index(drop=True), df_pc.reset_index(drop=True)], axis=1)

        return df

    def __split_data(self):
        train_ratio = self._preset.train_ratio
        print_verbose(f"Splitting data into {train_ratio*100:.1f}% train and {(1-train_ratio)*100:.1f}% validation sets...")
        train_df = self._df.sample(frac=train_ratio)
        validation_df = self._df.drop(train_df.index)
        print_verbose(f"train data count: {len(train_df)}")
        print_verbose(f"validation data count: {len(validation_df)}")
        return train_df, validation_df

    def __balance_data(self, train_df):
        """
        Upsample bug>0 rows in df until the ratio of bug=True to total is > balance_ratio.
        Returns the entire DataFrame (not just the training set).
        """
        balance_ratio = self._preset.balance_ratio
        print_verbose(f"Balancing data to have at most {balance_ratio*100:.1f}% of bug>0 samples in the training set...")
        bug_true = train_df[train_df[self._bug_tag] > 0]
        n_total = len(train_df)
        n_true = len(bug_true)
        ratio = n_true / n_total if n_total > 0 else 0
        print_verbose(f"Initial data: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
        if n_true > 0 and ratio <= balance_ratio:
            while True:
                n_true = len(train_df[train_df[self._bug_tag] > 0])
                n_total = len(train_df)
                ratio = n_true / n_total
                if ratio > balance_ratio:
                    break
                n_to_add = min(n_true, int(n_total * balance_ratio) - n_true)
                n_to_add = max(1, n_to_add)
                bug_true_upsampled = bug_true.sample(n=n_to_add, replace=True)
                train_df = pandas.concat([train_df, bug_true_upsampled], ignore_index=True)
            print_verbose(f"Data balanced: {n_total} samples ({n_true} bug>0, {n_total - n_true} bug=0, ratio={ratio:.2f})")
        else:
            print_verbose(f"Data already balanced")
        return train_df
    
    def __train_model(self, train_df):
        print_verbose("Training model...")
        input_columns = [col for col in train_df.columns if col not in [self._name_tag, self._bug_tag]]
        X_train = train_df[input_columns]
        y_train = train_df[self._bug_tag].astype(bool) if self._preset._use_boolean_model else train_df[self._bug_tag]
        clf = sklearn.ensemble.RandomForestClassifier()
        clf.fit(X_train, y_train)
        return clf, input_columns

    def __evaluate_model(self, clf, input_columns, validation_df):
        print_verbose("Evaluating model...")
        print_verbose("-----")
        print_verbose(f"Results for model ID '{self._rep_id}':")
        X_val = validation_df[input_columns]
        y_val = validation_df[self._bug_tag].astype(bool)
        y_pred = clf.predict(X_val).astype(bool)
        accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
        precision = sklearn.metrics.precision_score(y_val, y_pred, zero_division=0, average=None)
        recall = sklearn.metrics.recall_score(y_val, y_pred, zero_division=0, average=None)
        f1 = sklearn.metrics.f1_score(y_val, y_pred, zero_division=0, average=None)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
        print_verbose(f"Validation accuracy: {accuracy:.4f}")
        print_verbose(f"Validation precision (False, True): {precision}")
        print_verbose(f"Validation recall (False, True): {recall}")
        print_verbose(f"Validation F1-score (False, True): {f1}")
        print_verbose("Confusion matrix:")
        print_verbose(confusion_matrix)
        print_verbose("Classification report:")
        print_verbose(sklearn.metrics.classification_report(y_val, y_pred, zero_division=0))

        auc = None

        # Calculate AUC if possible
        try:
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(validation_df[input_columns])[:, 1]
                auc = sklearn.metrics.roc_auc_score(validation_df[self._bug_tag].astype(bool), y_proba)
                import matplotlib.pyplot as plt

                fpr, tpr, _ = sklearn.metrics.roc_curve(validation_df[self._bug_tag].astype(bool), y_proba)
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                plt.savefig(f'log/img/roc_curve_{self._rep_id}.png')
                plt.close()
                print_verbose(f"Validation AUC: {auc:.4f}")
            else:
                print("AUC cannot be calculated: classifier does not support predict_proba.")
        except Exception as e:
            print(f"Error calculating AUC: {e}")

        return accuracy, precision, recall, f1, confusion_matrix, auc


def print_data_stats(df, tags, bug_tag):
    data_size = len(df)
    bug_module_count = len(df[df[bug_tag] == True])
    print_verbose(f"metrics: {tags}")
    print_verbose(f"data count: {data_size}")
    print_verbose(f"modules with bugs: {bug_module_count} ({(bug_module_count / data_size) * 100:.2f}%)")



def main():
    try:
        for test in Runner.iter_tests():
            for model in test:
                model.run()
                model.save_results()
            test.save_results()

        print("\nAll tests completed.")

    # Ctrl C
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    #except Exception:
    #    print(f"\nAn error occurred: {traceback.format_exc()}")
    #    print(f"Ran {test_id} of {number_of_tests} tests.")
    finally:
        Test.writer().close()
        Model.writer().close()

if __name__ == "__main__":
    main()
