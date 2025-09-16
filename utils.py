import abc

class HashableElement(abc.ABC):
    all_elements = None

    def __new__(cls, *args, **kwargs):
        if cls.all_elements is None:
            cls.all_elements = {}

        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)

        if obj.name in cls.all_elements:
            raise ValueError(f"'{obj.name}' already exists.")
        cls.all_elements[obj.name] = obj

        return obj

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash((self.__class__.__module__, self.__class__.__name__, self._name.upper()))

    @classmethod
    def from_name(cls, name):
        if cls.all_elements is None or name not in cls.all_elements:
            raise ValueError(f"'{name}' does not exist.")
        return cls.all_elements[name]
    
    @classmethod
    def from_names(cls, names):
        return [cls.from_name(name) for name in names] if names else cls.all()

    @classmethod
    def all(cls):
        if cls.all_elements is None:
            return []
        return list(cls.all_elements.values())

    @classmethod
    def all_names(cls):
        if cls.all_elements is None:
            return []
        return list(cls.all_elements.keys())
