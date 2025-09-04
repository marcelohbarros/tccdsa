import csv
import pathlib
import atexit

class CsvWriter:
    def __init__(self, file_path):
        print(f"Creating csv file to save the results: {file_path}")
        self._path = pathlib.Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open(mode='w', newline='')
        self._field_names = CsvRowData.get_csv_row_names()
        self._writer = csv.DictWriter(self._file, fieldnames=self._field_names)
        self._writer.writeheader()
        self._is_open = True
        atexit.register(self.close)

    def write(self, data):
        if set(data.keys()) != set(self._field_names):
            raise ValueError("Data keys do not match the field names")
        if not self._is_open:
            raise ValueError("The file is already closed")
        self._writer.writerow(data)

    def close(self):
        print(f"Closing CSV file: {self._path}")        
        if self._is_open:
            self._file.close()
            self._is_open = False


class CsvRowData:
    _input_format = [
        'dataset',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'confusion_matrix'
    ]

    _conversion = {
        'dataset': lambda x: x['dataset'],
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
    }

    @classmethod
    def get_csv_row_names(cls):
        return list(cls._conversion.keys())

    def __init__(self, *inputs):
        if len(inputs) != len(self._input_format):
            raise ValueError(f"Expected {len(self._input_format)} inputs, got {len(inputs)}")

        self._inputs = {name: value for name, value in zip(self._input_format, inputs)}
        self._data = {name: self._conversion[name](self._inputs) for name in self._conversion.keys()}

    def to_dict(self):
        return self._data