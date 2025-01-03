import pandas as pd
import numpy as np
from script.data_variables.variables import Variables
from script.big_data.sparkconnection import SparkConnection
from script.logger import Log
import os

class BBGDataPreparation:

    def __init__(self):

        self.fs_path = '../../data/input/union_ebitda_rev_cashflowfromoper_capex_merged_with_x_vars.csv'
        self.comp_path = '../../data/input/Company Info.xlsx'

        self.x_cols_to_process = Variables().X_SELECTED
        self.industry_cols = Variables().INDUSTRY
        sp_conn = SparkConnection()
        self.spark = sp_conn.get_spark_conn('DataPreparation')
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()

    def read_csv(self):

        self.logger.info('loading x and y data...')
        ################################# FILE 1 #################################
        # Reading the CSV file into a pandas DataFrame
        df = pd.read_csv(self.fs_path)
        # Converting the 'FUNDAMENTAL_UPDATE_DT' column to a datetime format (assuming dates are in YYYYMMDD format)
        df['FUNDAMENTAL_UPDATE_DT'] = pd.to_datetime(df['FUNDAMENTAL_UPDATE_DT'], format='%Y%m%d')

        # Filtering the DataFrame to include only rows with "Annual" fiscal year periods.
        # This is determined by checking if the 'FISCAL_YEAR_PERIOD' column ends with " A".
        df_annual = df[df["FISCAL_YEAR_PERIOD"].str.endswith(" A")]

        # Extracting the year from the 'FISCAL_YEAR_PERIOD' column (e.g., "2023 A" -> 2023)
        # Adding the extracted year as a new column called 'Year'
        df_annual['Year'] = df_annual['FISCAL_YEAR_PERIOD'].str.extract(r'(\d{4})').astype(int)

        # Filtering the DataFrame to include only rows where the year is 2000 or later
        df_annual_after_2000 = df_annual[df_annual['Year'] >= 2000]

        # Sorting the filtered DataFrame by 'TICKER' (company identifier) and 'Year' in ascending order
        # Resetting the index of the sorted DataFrame
        df_annual_sorted_after_2000 = df_annual_after_2000.sort_values(by=['TICKER', 'Year']).reset_index(drop=True)

        self.logger.info('loading industry data...')
        ################################# FILE 2 #################################
        # Reading the Excel file containing industry code mappings into a DataFrame
        # Assuming the sheet 'industry code mapping' contains the relevant mappings
        df_industry_code_mapping = pd.read_excel(self.comp_path, sheet_name='industry code mapping')

        # Creating dictionaries to map numeric codes to descriptive industry labels for various industry levels
        industry_sector_mapping = dict(
            zip(df_industry_code_mapping['Industry_sector_num'], df_industry_code_mapping['Industry_sector']))
        industry_group_mapping = dict(
            zip(df_industry_code_mapping['Industry_group_num'], df_industry_code_mapping['Industry_group']))
        industry_subgroup_mapping = dict(
            zip(df_industry_code_mapping['Industry_subgroup_num'], df_industry_code_mapping['Industry_subgroup']))
        industry_level4_mapping = dict(
            zip(df_industry_code_mapping['Industry_level_4_num'], df_industry_code_mapping['Industry_level_4']))
        industry_level5_mapping = dict(
            zip(df_industry_code_mapping['Industry_level_5_num'], df_industry_code_mapping['Industry_level_5']))
        industry_level6_mapping = dict(
            zip(df_industry_code_mapping['Industry_level_6_num'], df_industry_code_mapping['Industry_level_6']))
        # industry_level7_mapping = dict(
        #     zip(df_industry_code_mapping['Industry_level_7_num'], df_industry_code_mapping['Industry_level_7']))

        # Storing all the mapping dictionaries in a list for easier handling or iteration
        industry_mappings = [
            industry_sector_mapping,
            industry_group_mapping,
            industry_subgroup_mapping,
            industry_level4_mapping,
            industry_level5_mapping,
            industry_level6_mapping
        ]
        self.logger.info('loading company data...')
        ################################# FILE 3 #################################
        # Reading the 'company info' sheet from the Excel file into a DataFrame
        df_company_info = pd.read_excel(self.comp_path, sheet_name='company info')

        # Extracting the ticker symbol from the 'Ticker' column
        # Assuming the ticker symbol is followed by " US" (e.g., "AAPL US"), and we need only the symbol part
        df_company_info['TICKER'] = df_company_info['Ticker'].str.extract(r'(.*) US')

        return (df_annual_sorted_after_2000, industry_mappings, df_company_info)

    def merge_fs_compinfo(self,
                          df_annual_sorted_after_2000,
                          industry_mappings,
                          df_company_info):
        self.logger.info('XY and Industry -> merged data')
        industry_cols_to_merge = []  # List to store names of the new mapped columns

        # Iterate over each column and its corresponding mapping
        for col, mapping in zip(self.industry_cols, industry_mappings):
            new_col_name = col + '_mapped'  # Generate new column name
            industry_cols_to_merge.append(new_col_name)  # Add new column name to the list

            # Create the new column by mapping values using the provided dictionary
            df_company_info[new_col_name] = df_company_info[col].map(mapping)

        # Combine the industry mapped columns and the 'TICKER' column into a list
        columns_to_select = industry_cols_to_merge + ['TICKER']

        # Perform a left join to merge the two DataFrames on the 'TICKER' column
        df_merged = pd.merge(
            df_annual_sorted_after_2000,
            df_company_info[columns_to_select],
            on='TICKER',
            how='left'
        )

        # Display the merged DataFrame
        return df_merged, industry_cols_to_merge

if __name__ == '__main__':
    dataprocessing = BBGDataPreparation()
    (df_annual_sorted_after_2000, industry_mappings, df_company_info) = dataprocessing.read_csv()
    df_merged = dataprocessing.merge_fs_compinfo(df_annual_sorted_after_2000, industry_mappings, df_company_info)
    print(df_merged)
    # Specifying the list of columns (variables) related to financial metrics and indicators for further processing