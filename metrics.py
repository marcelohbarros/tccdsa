import csv

import utils

class Metric(utils.HashableElement):
    all_tags = None

    def __new__(cls, *args, **kwargs):
        if cls.all_tags is None:
            cls.all_tags = set()

        tags = [tag.lower() for tag in args[2]]

        for tag in tags:
            if tag in cls.all_tags:
                raise ValueError(f"Tag '{tag}' already exists in Metric class.")

        obj = super().__new__(cls, *args, **kwargs)
        cls.all_tags.update({tag.lower() for tag in tags})
        return obj

    @classmethod
    def from_tag(cls, tag):
        tag = tag.lower()
        for m in cls.all():
            if tag in m.tags:
                return m

    def __init__(self, name, description, tags):
        super().__init__(name, description)
        self._tags = {tag.lower() for tag in tags}

    @property
    def tags(self):
        return self._tags

    def __repr__(self):
        return f"Metric(\"{self._name}\", \"{self._description}\", [\"{'", "'.join(self._tags)}\"])"


# Obligatory columns
NAME = Metric("NAME", "Name", ["name", "SHA"])
BUG = Metric("BUG", "Bug count", ["bug", "defect"])

# Metrics
AMC = Metric("AMC", "Average Method Complexity", ["amc"])
ASSIGNMENT_COUNT = Metric("ASSIGNMENT_COUNT", "Number of Assignments", ["assignmentsQty"])
CA = Metric("CA", "Afferent Coupling", ["ca"])
CAM = Metric("CAM", "Cohesion Among Methods", ["cam"])
CBM = Metric("CBM", "Coupling Between Methods", ["cbm"])
CBO = Metric("CBO", "Coupling Between Objects/CountClassCoupled", ["cbo"])
CC_AVG = Metric("CC_AVG", "Average Cyclomatic Complexity", ["avg_cc"])
CC_MAX = Metric("CC_MAX", "Maximum Cyclomatic Complexity", ["max_cc"])
CE = Metric("CE", "Efferent Couplings", ["ce"])
COMPARISON_COUNT = Metric("COMPARISON_COUNT", "Number of Comparisons", ["comparisonsQty"])
DAM = Metric("DAM", "Data Access Method", ["dam"])
DIT = Metric("DIT", "Depth of Inheritance Tree", ["dit"])
FIELD_COUNT = Metric("FIELD_COUNT", "Number of Fields", ["totalFields"])
IC = Metric("IC", "Inheritance Coupling", ["ic"])
LCOM = Metric("LCOM", "Lack of Cohesion in Methods", ["lcom"])
LCOM3 = Metric("LCOM3", "Lack of Cohesion in Methods 3", ["lcom3"])
LOC = Metric("LOC", "Lines of Code", ["loc", "loc_total", "iocode", "nloc", "countLineCode", "sloc", "nocl", "number_of_lines"])
LOOP_COUNT = Metric("LOOP_COUNT", "Number of Loops", ["loopQty"])
MATH_OPERATION_COUNT = Metric("MATH_OPERATION_COUNT", "Number of Math Operations", ["mathOperationsQty"])
METHOD_COUNT = Metric("METHOD_COUNT", "Number of Methods", ["nom", "totalMethods", "numberOfMethods"])
MFA = Metric("MFA", "Measure of Functional Abstraction", ["mfa"])
MOA = Metric("MOA", "Measure of Aggregation", ["moa"])
NESTED_BLOCKS_MAX = Metric("NESTED_BLOCKS_MAX", "Maximum Nested Blocks", ["maxNestedBlocks"])
NOC = Metric("NOC", "Number of Children", ["noc", "nsub"])
NPM = Metric("NPM", "Number of Public Methods / Class Interface Size", ["npm", "nopm", "cis", "npbm"])
NUMERIC_LITERAL_COUNT = Metric("NUMERIC_LITERAL_COUNT", "Number of Numeric Literals", ["numbersQty"])
PARENTHESIS_EXPRESSION_COUNT = Metric("PARENTHESIS_EXPRESSION_COUNT", "Number of Parenthesis Expressions", ["parenthesizedExpsQty"])
RETURN_COUNT = Metric("RETURN_COUNT", "Number of Returns", ["returnsQty"])
RFC = Metric("RFC", "Response for a Class", ["rfc"])
STATIC_INVOCATION_COUNT = Metric("STATIC_INVOCATION_COUNT", "Number of Static Invocations", ["nosi"])
STRING_LITERAL_COUNT = Metric("STRING_LITERAL_COUNT", "Number of String Literals", ["stringLiteralsQty"])
TRY_CATCH_COUNT = Metric("TRY_CATCH_COUNT", "Number of Try-Catch Blocks", ["tryCatchQty"])
UNIQUE_WORD_COUNT = Metric("UNIQUE_WORD_COUNT", "Number of Unique Words", ["uniqueWordsQty"])
VARIABLE_COUNT = Metric("VARIABLE_COUNT", "Number of Variables", ["variablesQty"])
VERSION = Metric("VERSION", "Project version", ["version"])
WMC = Metric("WMC", "Weighted Methods per Class", ["wmc"])


class MetricTag:
    def __init__(self, tag):
        self._metric = Metric.from_tag(tag)
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
