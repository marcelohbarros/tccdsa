import abc
import csv
import pathlib

from log import print_verbose

class CsvWriter(abc.ABC):
    _file = None
    _row_data_class = None

    def __init__(self):
        if self._file is None or self._row_data_class is None:
            raise NotImplementedError("Subclasses must define _file and _row_data_class")
        print_verbose(f"Creating csv file to save the results: {self._file}")
        self._path = pathlib.Path(self._file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open(mode='w', newline='')
        self._field_names = self._row_data_class.get_csv_row_names()
        self._writer = csv.DictWriter(self._file, fieldnames=self._field_names)
        self._writer.writeheader()
        self._is_open = True

    def write(self, data):
        row = self._row_data_class(*data).to_dict()
        if set(row.keys()) != set(self._field_names):
            raise ValueError("Data keys do not match the field names")
        if not self._is_open:
            raise ValueError("The file is already closed")
        self._writer.writerow(row)

    def close(self):
        print(f"Closing CSV file: {self._path}")        
        if self._is_open:
            self._file.close()
            self._is_open = False


class CsvRowData(abc.ABC):
    _input_format = None
    _conversion = None

    @classmethod
    def get_csv_row_names(cls):
        if cls._conversion is None:
            raise NotImplementedError("Subclasses must define _conversion")
        return list(cls._conversion.keys())

    def __init__(self, *inputs):
        if self._input_format is None or self._conversion is None:
            raise NotImplementedError("Subclasses must define _input_format and _conversion")
        if len(inputs) != len(self._input_format):
            raise ValueError(f"Expected {len(self._input_format)} inputs, got {len(inputs)}")

        self._inputs = {name: value for name, value in zip(self._input_format, inputs)}
        self._data = {name: self._conversion[name](self._inputs) for name in self._conversion.keys()}

    def to_dict(self):
        return self._data
    

class ModelCsvRowData(CsvRowData):
    _input_format = [
        'id',
        'test_id',
        'number_of_features',
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
        'number_of_features': lambda x: int(x['number_of_features']),
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


class TestCsvRowData(CsvRowData):
    _input_format = [
        'id',
        'dataset',
        'balance_ratio',
        'use_boolean_model'
    ]

    _conversion = {
        'id': lambda x: x['id'],
        'dataset': lambda x: x['dataset'],
        'balance_ratio': lambda x: float(x['balance_ratio']),
        'use_boolean_model': lambda x: bool(x['use_boolean_model'])
    }


class ModelCsvWriter(CsvWriter):
    _file = 'log/models.csv'
    _row_data_class = ModelCsvRowData

class TestCsvWriter(CsvWriter):
    _file = 'log/tests.csv'
    _row_data_class = TestCsvRowData
