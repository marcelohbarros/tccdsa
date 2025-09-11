class Default:
    METRICS = None
    TRAIN_LEN = 0.7
    BALANCE_RATIO = 0.5
    USE_BOOLEAN_MODEL = False
    PCA_FEATURES = None

class PreSet:
    all_presets = None

    def __new__(cls, *args, **kwargs):
        if cls.all_presets is None:
            cls.all_presets = {}

        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)

        if obj.name in cls.all_presets:
            raise ValueError(f"PreSet '{obj.name}' already exists.")
        cls.all_presets[obj.name] = obj

        return obj

    @classmethod
    def preset_from_name(cls, name):
        if cls.all_presets is None or name not in cls.all_presets:
            raise ValueError(f"PreSet '{name}' does not exist.")
        return cls.all_presets[name]

    @classmethod
    def preset_from_names(cls, names):
        return [cls.preset_from_name(name) for name in names] if names else cls.get_all_presets()

    @classmethod
    def get_all_presets(cls):
        if cls.all_presets is None:
            return []
        return list(cls.all_presets.values())

    @classmethod
    def all_preset_names(cls):
        if cls.all_presets is None:
            return []
        return list(cls.all_presets.keys())

    def __init__(self, name, description, metrics=Default.METRICS, train_len=Default.TRAIN_LEN,
                 balance_ratio=Default.BALANCE_RATIO, use_boolean_model=Default.USE_BOOLEAN_MODEL,
                 pca_features=Default.PCA_FEATURES):
        self._name = name
        self._description = description
        self._metrics = metrics if metrics is not None else []
        self._train_len = train_len
        self._balance_ratio = balance_ratio
        self._use_boolean_model = use_boolean_model
        self._pca_features = pca_features

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def metrics(self):
        return self._metrics

    @property
    def train_len(self):
        return self._train_len

    @property
    def balance_ratio(self):
        return self._balance_ratio

    @property
    def use_boolean_model(self):
        return self._use_boolean_model

    @property
    def pca_features(self):
        return self._pca_features

    def __repr__(self):
        return f"PreSet(name=\"{self._name}\", description=\"{self._description}\", metrics={self._metrics}, train_len={self._train_len}, balance_ratio={self._balance_ratio}, use_boolean_model={self._use_boolean_model}, pca_features={self._pca_features})"

    def __hash__(self):
        return hash((self._name,))
    

DEFAULT = PreSet("default", "Default settings")
