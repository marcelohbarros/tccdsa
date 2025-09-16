import utils

class Default:
    METRICS = None
    TRAIN_LEN = 0.7
    BALANCE_RATIO = 0.5
    USE_BOOLEAN_MODEL = False
    PCA_FEATURES = None

class PreSet(utils.HashableElement):
    all_presets = None

    def __init__(self, name, description, metrics=Default.METRICS, train_len=Default.TRAIN_LEN,
                 balance_ratio=Default.BALANCE_RATIO, use_boolean_model=Default.USE_BOOLEAN_MODEL,
                 pca_features=Default.PCA_FEATURES):
        super().__init__(name)
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


DEFAULT = PreSet("default", "Default settings")
BALANCE_10 = PreSet("balance_01", "Balance ratio 0.1", balance_ratio=0.1)
BALANCE_20 = PreSet("balance_02", "Balance ratio 0.2", balance_ratio=0.2)
BALANCE_30 = PreSet("balance_03", "Balance ratio 0.3", balance_ratio=0.3)
BALANCE_40 = PreSet("balance_04", "Balance ratio 0.4", balance_ratio=0.4)
TRAIN_50 = PreSet("train_05", "Training size 50%", train_len=0.5)
TRAIN_60 = PreSet("train_06", "Training size 60%", train_len=0.6)
TRAIN_80 = PreSet("train_08", "Training size 80%", train_len=0.8)
PCA_5 = PreSet("pca_05", "PCA with 5 features", pca_features=5)
PCA_10 = PreSet("pca_10", "PCA with 10 features", pca_features=10)
PCA_15 = PreSet("pca_15", "PCA with 15 features", pca_features=15)
PCA_20 = PreSet("pca_20", "PCA with 20 features", pca_features=20)
PCA_25 = PreSet("pca_25", "PCA with 25 features", pca_features=25)

DEFAULT_BOOLEAN = PreSet("boolean", "Use boolean model", use_boolean_model=True)
BALANCE_10_BOOLEAN = PreSet("balance_01_boolean", "Balance ratio 0.1 with boolean model", balance_ratio=0.1, use_boolean_model=True)
BALANCE_20_BOOLEAN = PreSet("balance_02_boolean", "Balance ratio 0.2 with boolean model", balance_ratio=0.2, use_boolean_model=True)
BALANCE_30_BOOLEAN = PreSet("balance_03_boolean", "Balance ratio 0.3 with boolean model", balance_ratio=0.3, use_boolean_model=True)
BALANCE_40_BOOLEAN = PreSet("balance_04_boolean", "Balance ratio 0.4 with boolean model", balance_ratio=0.4, use_boolean_model=True)
TRAIN_50_BOOLEAN = PreSet("train_05_boolean", "Training size 50% with boolean model", train_len=0.5, use_boolean_model=True)
TRAIN_60_BOOLEAN = PreSet("train_06_boolean", "Training size 60% with boolean model", train_len=0.6, use_boolean_model=True)
TRAIN_80_BOOLEAN = PreSet("train_08_boolean", "Training size 80% with boolean model", train_len=0.8, use_boolean_model=True)
PCA_5_BOOLEAN = PreSet("pca_05_boolean", "PCA with 5 features and boolean model", pca_features=5, use_boolean_model=True)
PCA_10_BOOLEAN = PreSet("pca_10_boolean", "PCA with 10 features and boolean model", pca_features=10, use_boolean_model=True)
PCA_15_BOOLEAN = PreSet("pca_15_boolean", "PCA with 15 features and boolean model", pca_features=15, use_boolean_model=True)
PCA_20_BOOLEAN = PreSet("pca_20_boolean", "PCA with 20 features and boolean model", pca_features=20, use_boolean_model=True)
PCA_25_BOOLEAN = PreSet("pca_25_boolean", "PCA with 25 features and boolean model", pca_features=25, use_boolean_model=True)

