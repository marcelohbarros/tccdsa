import abc
import csv
import pathlib
import statistics

import pandas

import metrics as m
from log import print_verbose

class CsvWriter(abc.ABC):
    _row_data_class = None

    def __init__(self):
        if self._row_data_class is None:
            raise NotImplementedError("Subclasses must define _file and _row_data_class")
        print_verbose(f"Creating csv file to save the results: {self._row_data_class._file}")
        self._path = pathlib.Path(self._row_data_class._file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open(mode='w', newline='')
        self._field_names = self._row_data_class.get_csv_row_names()
        self._writer = csv.DictWriter(self._file, fieldnames=self._field_names, delimiter=';')
        self._writer.writeheader()
        self._is_open = True

    def write(self, data):
        row = self._row_data_class(*data).to_dict()
        if set(row.keys()) != set(self._field_names):
            raise ValueError("Data keys do not match the field names")
        if not self._is_open:
            raise ValueError("The file is already closed")
        self._writer.writerow(row)
        return row

    def close(self):
        print(f"Closing CSV file: {self._path}")        
        if self._is_open:
            self._file.close()
            self._is_open = False


class CsvRowData(abc.ABC):
    _file = None
    _input_format = None
    _conversion = None

    @classmethod
    def get_csv_row_names(cls):
        if cls._conversion is None:
            raise NotImplementedError("Subclasses must define _conversion")
        return list(cls._conversion.keys())

    def __init__(self, *inputs):
        if self._input_format is None or self._conversion is None or self._file is None:
            raise NotImplementedError("Subclasses must define _input_format, _conversion and _file")
        if len(inputs) != len(self._input_format):
            raise ValueError(f"Expected {len(self._input_format)} inputs, got {len(inputs)}")

        self._inputs = {name: value for name, value in zip(self._input_format, inputs)}
        self._data = {name: self._conversion[name](self._inputs) for name in self._conversion.keys()}

    def to_dict(self):
        return self._data
    

class ModelCsvRowData(CsvRowData):
    _file = 'log/models.csv'

    _input_format = [
        'id',
        'test_id',
        'number_of_inputs',
        'duration',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'confusion_matrix',
        'auc'
    ]

    _conversion = {
        'id': lambda x: x['id'],
        'test_id': lambda x: x['test_id'],
        'number_of_inputs': lambda x: int(x['number_of_inputs']),
        'duration': lambda x: float(x['duration']),
        'accuracy': lambda x: x['accuracy'],
        'precision_true': lambda x: float(x['precision'][1]),
        'precision_false': lambda x: float(x['precision'][0]),
        'recall_true': lambda x: float(x['recall'][1]),
        'recall_false': lambda x: float(x['recall'][0]),
        'f1_true': lambda x: float(x['f1'][1]),
        'f1_false': lambda x: float(x['f1'][0]),
        'true_positive': lambda x: int(x['confusion_matrix'][1, 1]),
        'true_negative': lambda x: int(x['confusion_matrix'][0, 0]),
        'false_positive': lambda x: int(x['confusion_matrix'][0, 1]),
        'false_negative': lambda x: int(x['confusion_matrix'][1, 0]),
        'auc': lambda x: float(x['auc']) if x['auc'] is not None else None
    }


def model_data_max(key):
    def extractor(data):
        return max(model_data[key] for model_data in data['model_data'])
    return extractor


def model_data_min(key):
    def extractor(data):
        return min(model_data[key] for model_data in data['model_data'])
    return extractor


def model_data_avg(key):
    def extractor(data):
        return statistics.fmean(model_data[key] for model_data in data['model_data'])
    return extractor


def model_data_median(key):
    def extractor(data):
        return statistics.median(model_data[key] for model_data in data['model_data'])
    return extractor


def model_data_stdev(key):
    def extractor(data):
        return statistics.stdev(model_data[key] for model_data in data['model_data'])
    return extractor


class TestCsvRowData(CsvRowData):
    _file = 'log/tests.csv'

    _input_format = [
        'id',
        'dataset',
        'preset_name',
        'duration',
        'train_ratio',
        'balance_ratio',
        'use_boolean_model',
        'pca_number_of_features',
        'estimators',
        'criterion',
        'max_features',
        'bootstrap',
        'data_len',
        'model_data'
    ]

    _conversion = {
        'id': lambda x: x['id'],
        'dataset': lambda x: x['dataset'],
        'preset_name': lambda x: x['preset_name'],
        'duration': lambda x: float(x['duration']),
        'train_ratio': lambda x: float(x['train_ratio']),
        'balance_ratio': lambda x: float(x['balance_ratio']),
        'use_boolean_model': lambda x: bool(x['use_boolean_model']),
        'pca_number_of_features': lambda x: int(x['pca_number_of_features']) if x['pca_number_of_features'] is not None else None,
        'estimators': lambda x: int(x['estimators']),
        'criterion': lambda x: x['criterion'],
        'max_features': lambda x: x['max_features'] if x['max_features'] is not None else None,
        'bootstrap': lambda x: bool(x['bootstrap']),
        'data_len': lambda x: int(x['data_len']),
        'max_accuracy': model_data_max('accuracy'),
        'min_accuracy': model_data_min('accuracy'),
        'avg_accuracy': model_data_avg('accuracy'),
        'median_accuracy': model_data_median('accuracy'),
        'min_precision_true': model_data_min('precision_true'),
        'max_precision_true': model_data_max('precision_true'),
        'median_precision_true': model_data_median('precision_true'),
        'min_precision_false': model_data_min('precision_false'),
        'max_precision_false': model_data_max('precision_false'),
        'median_precision_false': model_data_median('precision_false'),
        'min_recall_true': model_data_min('recall_true'),
        'max_recall_true': model_data_max('recall_true'),
        'median_recall_true': model_data_median('recall_true'),
        'min_recall_false': model_data_min('recall_false'),
        'max_recall_false': model_data_max('recall_false'),
        'median_recall_false': model_data_median('recall_false'),
        'median_f1_true': model_data_median('f1_true'),
        'median_f1_false': model_data_median('f1_false'),
        'median_true_positive': model_data_median('true_positive'),
        'median_true_negative': model_data_median('true_negative'),
        'median_false_positive': model_data_median('false_positive'),
        'median_false_negative': model_data_median('false_negative'),
        'min_auc': model_data_min('auc'),
        'max_auc': model_data_max('auc'),
        'avg_auc': model_data_avg('auc'),
        'median_auc': model_data_median('auc')
    }


class ModelCsvWriter(CsvWriter):
    _row_data_class = ModelCsvRowData


class TestCsvWriter(CsvWriter):
    _row_data_class = TestCsvRowData


class CsvReader(abc.ABC):
    _row_data_class = None

    def __init__(self):
        if self._row_data_class is None:
            raise NotImplementedError("Subclasses must define _file and _row_data_class")
        if not pathlib.Path(self._row_data_class._file).exists():
            raise FileNotFoundError(f"CSV file not found: {self._file}")

        self._df = pandas.read_csv(self._row_data_class._file, delimiter=';')

        if set(self._df.columns) != set(self._row_data_class.get_csv_row_names()):
            raise ValueError("CSV columns do not match the expected field names")
    
    @property
    def df(self):
        return self._df


class ModelCsvReader(CsvReader):
    _row_data_class = ModelCsvRowData


class TestCsvReader(CsvReader):
    _row_data_class = TestCsvRowData


class FileMetrics:
    def __init__(self, data_path):
        with open(data_path) as file:
            reader = csv.reader(file)
            file_column_tags = next(reader)

        self._metrics = {m.MetricTag(tag) for tag in file_column_tags}
        remove = {metric for metric in self._metrics if not metric.is_valid()}
        self._metrics.difference_update(remove)

        if self.get_name_tag() is None:
            raise ValueError("A name column is required in the CSV file.")
        if self.get_bug_tag() is None:
            raise ValueError("A bug column is required in the CSV file.")

    def filter_tags(self, filter_metrics):
        if not filter_metrics or filter_metrics.is_empty():
            return self.get_all_tags()
        return [metric.tag for metric in self._metrics if metric.metric in filter_metrics.metrics]

    def get_all(self):
        return [metric.metric for metric in self._metrics]

    def get_tag(self, metric):
        for m in self._metrics:
            if m.metric == metric:
                return m.tag

    def get_all_tags(self):
        return [metric.tag for metric in self._metrics]

    def get_name_tag(self):
        return self.get_tag(m.NAME)
    
    def get_bug_tag(self):
        return self.get_tag(m.BUG)
