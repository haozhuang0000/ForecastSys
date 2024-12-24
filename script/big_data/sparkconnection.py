import os
from datetime import datetime
from pyspark.sql import SparkSession
from script.database.mysqlconnection import MysqlConfig
from script.logger.logger import Log

from dotenv import load_dotenv
load_dotenv()

class SparkConnection:

    def __init__(self):
        self.logger = Log(f"{os.path.basename(__file__)}").getlog()
        self.spark_master_url = os.getenv('SPARK_MASTER_URL')
        if not self.spark_master_url:
            self.logger.info("ERROR: SPARK_MASTER_URL is not set in the environment variables.")
            raise ValueError("SPARK_MASTER_URL is required.")

    def get_spark_conn(self, appName: str) -> SparkSession:
        """
        Creates and returns a Spark session.

        Args:
            appName (str): Name of the Spark application.

        Returns:
            SparkSession: An active Spark session.
        """
        if not appName:
            raise ValueError("appName cannot be empty.")

        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_app_name = f"{appName}_{start_time}"
        self.logger.info(f"creating spark session with appName: {full_app_name}")

        try:
            spark = SparkSession.builder \
                .appName(full_app_name) \
                .remote(self.spark_master_url) \
                .getOrCreate()
            self.logger.info("spark session created successfully")
            return spark
        except Exception as e:
            self.logger.info(f"Error creating Spark session: \n {e}")
            raise

    def sp_get_data_from_mysql(self,
                               sparkSession: SparkSession,
                               db_name: str,
                               table: str):
        """
        Fetches data from a MySQL table and returns it as a Spark DataFrame.

        Args:
            sparkSession (SparkSession): An active Spark session.
            db_name (str): Name of the MySQL database.
            table (str): Name of the MySQL table.

        Returns:
            DataFrame: A Spark DataFrame containing the table data.
        """
        if not db_name or not table:
            raise ValueError("Both db_name and table must be provided.")

        url = f"jdbc:mysql://{MysqlConfig.MYSQL_HOST}:{MysqlConfig.MYSQL_PORT}/{db_name}"
        properties = {
            "user": MysqlConfig.MYSQL_USERNAME,
            "password": MysqlConfig.MYSQL_PASSWORD,
            "driver": "com.mysql.cj.jdbc.Driver"
        }

        self.logger.info(f"fetching data from MySQL: {db_name}.{table}")
        try:
            df = sparkSession.read.jdbc(url=url, table=table, properties=properties)
            self.logger.info(f"data fetched successfully from {db_name}.{table}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from MySQL: {e}")
            raise
