import abc

class HashableElement(abc.ABC):
    all_elements = None

    def __new__(cls, *args, **kwargs):
        if cls.all_elements is None:
            cls.all_elements = set()

        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)

        if obj in cls.all_elements:
            raise ValueError(f"'{obj.name}' already exists.")
        cls.all_elements.add(obj)

        return obj

    def __init__(self, name, description):
        self._name = name
        self._description = description

    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description

    def __hash__(self):
        return hash((self.__class__.__module__, self.__class__.__name__, self._name.upper()))

    @classmethod
    def from_name(cls, name):
        for element in cls.all_elements:
            if element.name == name:
                return element
        raise ValueError(f"'{name}' does not exist.")
    
    @classmethod
    def from_names(cls, names):
        return [cls.from_name(name) for name in names] if names else cls.all()

    @classmethod
    def all(cls):
        if cls.all_elements is None:
            return set()
        return cls.all_elements

    @classmethod
    def all_names(cls):
        if cls.all_elements is None:
            return []
        return [element.name for element in cls.all_elements]
