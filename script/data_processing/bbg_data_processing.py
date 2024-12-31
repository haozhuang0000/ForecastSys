import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from script.data_processing.bbg_data_preparation import BBGDataPreparation
from script.data_processing.bbg_data_aggregation import BBGDataAggregation
from script.data_processing.bbg_data_missingValueHandler import BBGDataMissingValueHandler
from script.logger.logger import Log

class BBGDataProcessing:
    def __init__(self):
        directory_path = "../../data/output"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        self.bbgdataprep = BBGDataPreparation()
        self.bbgdataagg = BBGDataAggregation()
        self.bbgdatamvhandler = BBGDataMissingValueHandler()
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()

    def prepare_train_data(self, df_combined):

        self.logger.info('preparing training_pipeline and test data - missing value handling')
        train_df_list = []
        ground_truth_list = []

        # Assuming cols_to_process is defined and contains the columns to be set to np.nan
        # cols_to_process = ['col1', 'col2', ...]
        self.logger.info('1. replaces columns where more than threshold of values are missing with 0 '
                         '2. imputes missing values in a pandas Series according to the specified rules '
                         '3. Combine processed training_pipeline data with modified test data')
        for id_bb_unique, sub_df in tqdm(df_combined.groupby('ID_BB_UNIQUE'),
                                   desc="Processing 1. replace 2. impute 3. combine"):
            # Calculate the number of rows to include (first 80%)
            n_rows = len(sub_df)
            n_train = int(n_rows * 0.8)

            # Split the data into training_pipeline and testing subsets
            train_sub_df = sub_df.iloc[:n_train]
            test_sub_df = sub_df.iloc[n_train:]

            # Store ground truth for the test set
            ground_truth_list.append(test_sub_df.copy())

            # Set specified columns in test_sub_df to np.nan
            test_sub_df[self.bbgdataprep.x_cols_to_process] = np.nan

            # Step 1: Drop columns with high missing values marked as np.nan
            df = self.bbgdatamvhandler.mark_high_missing_columns(train_sub_df, threshold=0.8)

            # Step 2: Handle missing values for each id_bb_unique
            df_processed = self.bbgdatamvhandler.impute_missing_values(df)

            # Combine processed training_pipeline data with modified test data
            df_train = pd.concat([df_processed, test_sub_df], ignore_index=True)

            # Add the combined data to the train list
            train_df_list.append(df_train)

        return train_df_list, ground_truth_list

    def saving_train_data(self, train_df_list, ground_truth_list):

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
        df_ticker_join = df_company_info[['ID_BB_UNIQUE', 'TICKER']]
        df_combined = df_combined.merge(df_ticker_join, how='left', on='ID_BB_UNIQUE')
        train_df_list, ground_truth_list = self.prepare_train_data(df_combined)
        self.saving_train_data(train_df_list, ground_truth_list)

if __name__ == '__main__':

    bbg_process = BBGDataProcessing()

    bbg_process.main()