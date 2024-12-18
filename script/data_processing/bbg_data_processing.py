from script.data_processing.bbg_data_preparation import BBGDataPreparation
from script.data_processing.bbg_data_aggregation import BBGDataAggregation
from script.data_processing.bbg_data_missingValueHandler import BBGDataMissingValueHandler
import pandas as pd
import numpy as np

class BBGDataProcessing:
    def __init__(self):
        self.bbgdataprep = BBGDataPreparation()
        self.bbgdataagg = BBGDataAggregation()
        self.bbgdatamvhandler = BBGDataMissingValueHandler()

    def prepare_train_data(self, df_combined):

        train_df_list = []
        ground_truth_list = []

        # Assuming cols_to_process is defined and contains the columns to be set to np.nan
        # cols_to_process = ['col1', 'col2', ...]

        for ticker, sub_df in df_combined.groupby('TICKER'):
            # Calculate the number of rows to include (first 80%)
            n_rows = len(sub_df)
            n_train = int(n_rows * 0.8)

            # Split the data into training and testing subsets
            train_sub_df = sub_df.iloc[:n_train]
            test_sub_df = sub_df.iloc[n_train:]

            # Store ground truth for the test set
            ground_truth_list.append(test_sub_df.copy())

            # Set specified columns in test_sub_df to np.nan
            test_sub_df[self.bbgdataprep.x_cols_to_process] = np.nan

            # Step 1: Drop columns with high missing values marked as np.nan
            df = self.bbgdatamvhandler.mark_high_missing_columns(train_sub_df, threshold=0.8)

            # Step 2: Handle missing values for each ticker
            df_processed = self.bbgdatamvhandler.impute_missing_values(df)

            # Combine processed training data with modified test data
            df_train = pd.concat([df_processed, test_sub_df], ignore_index=True)

            # Add the combined data to the train list
            train_df_list.append(df_train)

            print("-" * 30)
        return train_df_list, ground_truth_list

    def saving_train_data(self, train_df_list,ground_truth_list):
        df_combined_imputed = pd.concat(train_df_list, ignore_index=True)
        df_combined_imputed.to_csv('../../data/output/union_processed_imputed_80_20.csv', index=False)

        df_ground_truth = pd.concat(ground_truth_list, ignore_index=True)
        df_ground_truth.to_csv('../../data/output/union_processed_groundtruth_80_20.csv', index=False)

    def main(self):
        df_annual_sorted_after_2000, industry_mappings, df_company_info = self.bbgdataprep.read_csv()
        df_merged, industry_cols_to_merge = self.bbgdataprep.merge_fs_compinfo(
            df_annual_sorted_after_2000, industry_mappings, df_company_info
        )
        df_combined, cols_to_process = self.bbgdataagg.union_processing(df_merged, industry_cols_to_merge)
        train_df_list, ground_truth_list = self.prepare_train_data(df_combined)
        self.saving_train_data(train_df_list, ground_truth_list)

if __name__ == '__main__':

    bbg_process = BBGDataProcessing()

    bbg_process.main()