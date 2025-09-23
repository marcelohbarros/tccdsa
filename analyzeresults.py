import statistics
import pandas
import logging
import csvhandler as ch


logging.basicConfig(
    filename='log/analyzeresults.log',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'  # Overwrite the previous log
)


def log_info(message):
    print(message)
    logging.info(message)


def resume_variable_info(test_df, description, variable, filter_column=None, filter_value=None):
    median_column = f"median_{variable}"
    min_column = f"min_{variable}"
    max_column = f"max_{variable}"

    if filter_column is not None and filter_value is not None:
        test_df = test_df[test_df[filter_column] == filter_value]

    median_values = [value for value in test_df[median_column] if not pandas.isna(value)]
    min_values = [value for value in test_df[min_column] if not pandas.isna(value)]
    max_values = [value for value in test_df[max_column] if not pandas.isna(value)]

    return f"{description}\t{min(min_values):.4f}\t{statistics.median(median_values):.4f}\t{max(max_values):.4f}"


def print_report(test_df, model_df):
    test_size = len(test_df)
    presets = test_df['preset_name'].unique()
    datasets = test_df['dataset'].unique()
    log_info("==========================")
    log_info(f"Number of tests: {test_size}")
    log_info(f"Number of models: {len(model_df)}")
    log_info(f"Found {len(presets)} presets in the run: {', '.join(presets)}")
    log_info(f"Found {len(datasets)} datasets in the run: {', '.join(datasets)}")
    log_info("==========================")
    log_info("General results\t\tmin\tmedian\tmax")
    log_info(resume_variable_info(test_df, "Accuracy:\t", "accuracy"))
    log_info(resume_variable_info(test_df, "Precision true:\t", "precision_true"))
    log_info(resume_variable_info(test_df, "Precision false:", "precision_false"))
    log_info(resume_variable_info(test_df, "Recall true:\t", "recall_true"))
    log_info(resume_variable_info(test_df, "Recall false:\t", "recall_false"))
    #log_info(resume_variable_info(test_df, "F1 true", "f1_true"))
    #log_info(resume_variable_info(test_df, "F1 false", "f1_false"))
    log_info(resume_variable_info(test_df, "AUC:\t\t", "auc"))

    log_info("==========================")
    log_info("Presets:")
    for preset in presets:
        log_info("--------------------------")
        log_info(f"'{preset}'")
        log_info("\t\tmin\tmedian\tmax")
        log_info(resume_variable_info(test_df, "Accuracy:", "accuracy", "preset_name", preset))
        log_info(resume_variable_info(test_df, "Recall true:", "recall_true", "preset_name", preset))
        log_info(resume_variable_info(test_df, "Recall false:", "recall_false", "preset_name", preset))
        log_info(resume_variable_info(test_df, "AUC:\t", "auc", "preset_name", preset))

    log_info("==========================")
    log_info("Datasets:")
    for dataset in datasets:
        log_info("--------------------------")
        log_info(f"'{dataset}'")
        log_info("\t\tmin\tmedian\tmax")
        log_info(resume_variable_info(test_df, "Accuracy:", "accuracy", "dataset", dataset))
        log_info(resume_variable_info(test_df, "Recall true:", "recall_true", "dataset", dataset))
        log_info(resume_variable_info(test_df, "Recall false:", "recall_false", "dataset", dataset))
        log_info(resume_variable_info(test_df, "AUC:\t", "auc", "dataset", dataset))


def main():
    test_reader = ch.TestCsvReader()
    test_df = test_reader.df

    model_reader = ch.ModelCsvReader()
    model_df = model_reader.df

    print_report(test_df, model_df)


if __name__ == "__main__":
    main()