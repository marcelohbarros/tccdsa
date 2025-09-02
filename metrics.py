import csv

class Metric:
    all_tags = None
    all_metrics = None

    def __new__(cls, *args):
        if cls.all_tags is None:
            cls.all_tags = set()
        if cls.all_metrics is None:
            cls.all_metrics = []

        description = args[0]
        tags = [tag.lower() for tag in args[1]]

        for tag in tags:
            if tag in cls.all_tags:
                raise ValueError(f"Tag '{tag}' already exists in Metric class.")

        cls.all_tags.update({tag.lower() for tag in tags})

        obj = super().__new__(cls)
        obj.__init__(description, tags)
        cls.all_metrics.append(obj)
        return obj

    @classmethod
    def get_metric(cls, tag):
        tag = tag.lower()
        for m in cls.all_metrics:
            if tag in m.tags:
                return m

    def __init__(self, description, tags):
        self._tags = {tag.lower() for tag in tags}
        self._description = description

    @property
    def tags(self):
        return self._tags
    
    @property
    def description(self):
        return self._description

    def __repr__(self):
        return f"Metric(\"{self._description}\", [\"{'", "'.join(self._tags)}\"])"

    def __hash__(self):
        return hash((", ".join(self._tags),))

# Obligatory columns
NAME = Metric("Name", ["name", "SHA"])
BUG = Metric("Bug count", ["bug", "defect"])

# Metrics
AMC = Metric("Average Method Complexity", ["amc"])
ASSIGNMENT_COUNT = Metric("Number of Assignments", ["assignmentsQty"])
CA = Metric("Afferent Coupling", ["ca"])
CAM = Metric("Cohesion Among Methods", ["cam"])
CBM = Metric("Coupling Between Methods", ["cbm"])
CBO = Metric("Coupling Between Objects/CountClassCoupled", ["cbo"])
CC_AVG = Metric("Average Cyclomatic Complexity", ["avg_cc"])
CC_MAX = Metric("Maximum Cyclomatic Complexity", ["max_cc"])
CE = Metric("Efferent Couplings", ["ce"])
COMPARISON_COUNT = Metric("Number of Comparisons", ["comparisonsQty"])
DAM = Metric("Data Access Method", ["dam"])
DIT = Metric("Depth of Inheritance Tree", ["dit"])
FIELD_COUNT = Metric("Number of Fields", ["totalFields"])
IC = Metric("Inheritance Coupling", ["ic"])
LCOM = Metric("Lack of Cohesion in Methods", ["lcom"])
LCOM3 = Metric("Lack of Cohesion in Methods 3", ["lcom3"])
LOC = Metric("Lines of Code", ["loc", "loc_total", "iocode", "nloc", "countLineCode", "sloc", "nocl", "number_of_lines"])
LOOP_COUNT = Metric("Number of Loops", ["loopQty"])
MATH_OPERATION_COUNT = Metric("Number of Math Operations", ["mathOperationsQty"])
METHOD_COUNT = Metric("Number of Methods", ["nom", "totalMethods", "numberOfMethods"])
MFA = Metric("Measure of Functional Abstraction", ["mfa"])
MOA = Metric("Measure of Aggregation", ["moa"])
NESTED_BLOCKS_MAX = Metric("Maximum Nested Blocks", ["maxNestedBlocks"])
NOC = Metric("Number of Children", ["noc", "nsub"])
NPM = Metric("Number of Public Methods / Class Interface Size", ["npm", "nopm", "cis", "npbm"])
NUMERIC_LITERAL_COUNT = Metric("Number of Numeric Literals", ["numbersQty"])
PARENTHESIS_EXPRESSION_COUNT = Metric("Number of Parenthesis Expressions", ["parenthesizedExpsQty"])
RETURN_COUNT = Metric("Number of Returns", ["returnsQty"])
RFC = Metric("Response for a Class", ["rfc"])
STATIC_INVOCATION_COUNT = Metric("Number of Static Invocations", ["nosi"])
STRING_LITERAL_COUNT = Metric("Number of String Literals", ["stringLiteralsQty"])
TRY_CATCH_COUNT = Metric("Number of Try-Catch Blocks", ["tryCatchQty"])
UNIQUE_WORD_COUNT = Metric("Number of Unique Words", ["uniqueWordsQty"])
VARIABLE_COUNT = Metric("Number of Variables", ["variablesQty"])
VERSION = Metric("Project version", ["version"])
WMC = Metric("Weighted Methods per Class", ["wmc"])


class ArgumentMetrics:
    def __init__(self, tags, validate=False):
        self._required_columns = {NAME, BUG}
        self._metrics = self._required_columns.copy()
        self._metrics.update({Metric.get_metric(tag) for tag in tags})
        if None in self._metrics:
            self._metrics.remove(None)
            if validate:
                raise ValueError(f"A metric was not found from the tags: '{tags}'")

    @property
    def metrics(self):
        return self._metrics

    def is_empty(self):
        return len(self._metrics) == len(self._required_columns)

    @property
    def required_columns(self):
        return self._required_columns


class MetricTag:
    def __init__(self, tag):
        self._metric = Metric.get_metric(tag)
        self._tag = tag

    def is_valid(self):
        return self.metric is not None

    @property
    def metric(self):
        return self._metric

    @property
    def tag(self):
        return self._tag

    def __hash__(self):
        return hash((self.tag,))


class FileMetrics:
    def __init__(self, data_path):
        with open(data_path) as file:
            reader = csv.reader(file)
            file_column_tags = next(reader)

        self._metrics = {MetricTag(tag) for tag in file_column_tags}
        remove = {metric for metric in self._metrics if not metric.is_valid()}
        self._metrics.difference_update(remove)

        if self.get_name_tag() is None:
            raise ValueError("A name column is required in the CSV file.")
        if self.get_bug_tag() is None:
            raise ValueError("A bug column is required in the CSV file.")

    def get_metric(self, tag):
        for metric in self._metrics:
            if metric.tag == tag:
                return metric.metric

    def get_metrics(self, *, tags=None, validate=False):
        if tags is None:
            return self.get_all_metrics()

        metrics = [self.get_metric(tag) for tag in tags]
        if validate and None in metrics:
            raise ValueError(f"A metric was not found from the tags: '{tags}'")

        return metrics

    def get_all_metrics(self):
        return [metric.metric for metric in self._metrics]

    def get_tag(self, metric, validate=False):
        for m in self._metrics:
            if m.metric == metric:
                return m.tag
            
        if validate:
            raise ValueError(f"A tag was not found from the metric: '{metric}' in the file columns: '{self.get_all_tags()}'")

    def get_tags(self, *, metrics=None, argument_metrics=None, validate=False):
        if not argument_metrics.is_empty() and argument_metrics is not None:
            metrics = argument_metrics.metrics

        if metrics is None:
            return self.get_all_tags()

        tags = [self.get_tag(metric, validate) for metric in metrics if self.get_tag(metric) is not None]
        return tags

    def get_all_tags(self):
        return [metric.tag for metric in self._metrics]

    def get_name_tag(self):
        return self.get_tag(NAME)
    
    def get_bug_tag(self):
        return self.get_tag(BUG)
