import metrics
import utils

class Default:
    METRICS = None
    TRAIN_RATIO = 0.7
    BALANCE_RATIO = 0.5
    USE_BOOLEAN_MODEL = False
    PCA_FEATURES = None
    ESTIMATORS = 100
    CRITERION = 'gini'
    MAX_FEATURES = 'sqrt'
    BOOTSTRAP = True

class PreSet(utils.HashableElement):
    _id_counter = 1
    all_presets = None

    def __init__(self, name, description, metrics=Default.METRICS, train_ratio=Default.TRAIN_RATIO,
                 balance_ratio=Default.BALANCE_RATIO, use_boolean_model=Default.USE_BOOLEAN_MODEL,
                 pca_features=Default.PCA_FEATURES, estimators=Default.ESTIMATORS,
                 criterion=Default.CRITERION, max_features=Default.MAX_FEATURES, bootstrap=Default.BOOTSTRAP):
        super().__init__(name, description)
        self._id = PreSet._id_counter
        PreSet._id_counter += 1
        self._metric_tags = metrics if metrics is not None else []
        self._metrics = PreSetMetrics(self._metric_tags)
        self._train_ratio = train_ratio
        self._balance_ratio = balance_ratio
        self._use_boolean_model = use_boolean_model
        self._pca_features = pca_features
        self._estimators = estimators
        self._criterion = criterion
        self._max_features = max_features
        self._bootstrap = bootstrap

    @property
    def id(self):
        return self._id

    @property
    def metrics(self):
        return self._metrics

    @property
    def train_ratio(self):
        return self._train_ratio

    @property
    def balance_ratio(self):
        return self._balance_ratio

    @property
    def use_boolean_model(self):
        return self._use_boolean_model

    @property
    def pca_features(self):
        return self._pca_features

    @property
    def estimators(self):
        return self._estimators

    @property
    def criterion(self):
        return self._criterion

    @property
    def max_features(self):
        return self._max_features

    @property
    def bootstrap(self):
        return self._bootstrap

    def __repr__(self):
        return (f"PreSet(id={self._id}, name=\"{self._name}\", description=\"{self._description}\", metrics={self._metric_tags}, "
                f"train_ratio={self._train_ratio}, balance_ratio={self._balance_ratio}, use_boolean_model={self._use_boolean_model}, "
                f"pca_features={self._pca_features}, estimators={self._estimators}, criterion={self._criterion}, "
                f"max_features={self._max_features}, bootstrap={self._bootstrap})")


class PreSetMetrics:
    def __init__(self, tags):
        self._required_columns = {metrics.NAME, metrics.BUG}
        self._metrics = self._required_columns.copy()
        self._metrics.update({metrics.Metric.from_tag(tag) for tag in tags})
        if None in self._metrics:
            self._metrics.remove(None)
            raise ValueError(f"A metric was not found from the tags: '{tags}'")

    @property
    def metrics(self):
        return self._metrics

    def is_empty(self):
        return len(self._metrics) == len(self._required_columns)

    @property
    def required_columns(self):
        return self._required_columns


pca_features_values = [None, 25]
boolean_model_values = [False, True]
estimator_values = [10, 50, 100, 500]
criterion_values = ['gini', 'entropy']
max_features_values = ['sqrt', None]
bootstrap_values = [True, False]

# Generate all combinations of presets
def generate_all_presets():
    presets = [PreSet("default", "Default")]
    for pca_features in pca_features_values:
        for use_boolean_model in boolean_model_values:
            for estimators in estimator_values:
                for criterion in criterion_values:
                    for max_features in max_features_values:
                        for bootstrap in bootstrap_values:
                            pca_features_str = f"pca-{pca_features}_" if pca_features is not None else ""
                            boolean_model_str = f"booleanModel_" if use_boolean_model else ""
                            bootstrap_str = "_bootstrap" if bootstrap else ""
                            name = (f"{pca_features_str}{boolean_model_str}"
                                    f"est-{estimators}_crit-{criterion}_maxf-{max_features}{bootstrap_str}")
                            description = (f"PCA: {pca_features}, Boolean Model: {use_boolean_model}, "
                                           f"Estimators: {estimators}, Criterion: {criterion}, "
                                           f"Max Features: {max_features}, Bootstrap: {bootstrap}")
                            preset = PreSet(name, description, pca_features=pca_features,
                                            use_boolean_model=use_boolean_model, estimators=estimators,
                                            criterion=criterion, max_features=max_features, bootstrap=bootstrap)
                            presets.append(preset)
    print("Generated", len(presets), "presets.")

    return presets

generate_all_presets()