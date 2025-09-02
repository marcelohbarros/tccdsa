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
        self.tags = {tag.lower() for tag in tags}
        self.description = description

    def __repr__(self):
        return f"Metric({self.name}, [\"{'", "'.join(self.tags)}\"])"

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


class MetricInstance:
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


class MetricSet:
    def __init__(self, tags):
        self._metrics = {MetricInstance(tag) for tag in tags} if tags else set()
  
    @property
    def metrics(self):
        return self._metrics
    
    @property
    def tags(self):
        return {m.tag for m in self._metrics}


class ArgumentMetricSet(MetricSet):
    def __init__(self, tags):
        super().__init__(tags)
        if None in self._metrics:
            raise ValueError(f"A metric was not found from the tags: '{tags}'")
        self._required_columns = {NAME, BUG}
        self._metrics.update(self._metrics)
  
    def is_empty(self):
        return bool(self._metrics)

    @property
    def required_columns(self):
        return self._required_columns


class CsvMetricSet(MetricSet):
    def __init__(self, data_path):
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)
            csv_file_columns = next(reader)
        super().__init__(csv_file_columns)

        remove = {metric for metric in self.metrics if not metric.is_valid()}
        self._metrics.difference_update(remove)
