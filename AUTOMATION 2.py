import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import schedule
import time
import logging
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pandas.api.types import is_numeric_dtype

# Configuration
PROCESSED_DIR = 'Processed_Database'
VIZ_DIR = 'Data_Visualization'

# Setup logging
logging.basicConfig(filename='automation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')  # 'w' to overwrite log each run

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def setup_directories():
    """Ensure necessary directories exist."""
    for directory in [PROCESSED_DIR, VIZ_DIR]:
        os.makedirs(directory, exist_ok=True)
    logging.info(f"Directories {PROCESSED_DIR} and {VIZ_DIR} setup.")


def fetch_data(database_url=None, file_path=r"C:\Users\amans\OneDrive\Documents\moviesdata.xlsx"):
    """
    Fetch data from either a database or a local file.

    :param database_url: URL string for database connection
    :param file_path: Path to the local file (CSV or Excel)
    :return: DataFrame or None if an error occurred
    """
    try:
        if file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
        elif database_url:
            engine = create_engine(database_url)
            Session = sessionmaker(bind=engine)
            with Session() as session:
                df = pd.read_sql_query('SELECT * FROM mytable', session.bind)
        else:
            raise ValueError("No data source specified")

        logging.info(f"Data fetched successfully. Shape: {df.shape}")
        return df.dropna()  # Simple data cleaning
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None


def analyze_data(df):
    """
    Perform some basic analysis on the data.

    :param df: DataFrame to analyze
    :return: Dictionary with analysis results
    """
    if df is None or df.empty:
        return {}

    analysis = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'summary': df.describe().to_dict()
    }
    return analysis


def visualize_data(df):
    """Create visualizations based on data types."""
    if df is None or df.empty:
        logging.warning("No data to visualize")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    if not numeric_cols.empty:  # Using .empty for Index objects
        # Histogram for the first numeric column if exists
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_cols[0]])
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(os.path.join(VIZ_DIR, 'histogram.png'))
        plt.close()
        logging.info(f"Histogram saved to {VIZ_DIR}/histogram.png")

        if len(numeric_cols) >= 2:
            # Scatter plot for the first two numeric columns
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            fig.write_html(os.path.join(VIZ_DIR, 'scatter.html'))
            logging.info(f"Scatter plot saved to {VIZ_DIR}/scatter.html")

    if not categorical_cols.empty:  # Using .empty for Index objects
        # Bar plot for the first categorical column if exists
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=categorical_cols[0])
        plt.title(f"Count of {categorical_cols[0]}")
        plt.savefig(os.path.join(VIZ_DIR, 'bar_plot.png'))
        plt.close()
        logging.info(f"Bar plot saved to {VIZ_DIR}/bar_plot.png")

def save_processed_data(df):
    """Save processed data to a CSV file."""
    if df is not None:
        csv_path = os.path.join(PROCESSED_DIR, 'processed_data.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Data saved to {csv_path}")


def job(database_url=None, file_path=None):
    """Main job function to process data."""
    setup_directories()
    df = fetch_data(database_url, file_path)
    if df is not None:
        analysis = analyze_data(df)
        logging.info("Data analysis completed")
        visualize_data(df)
        save_processed_data(df)
        logging.info("Job completed successfully")
    else:
        logging.warning("Job failed due to data issues")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data from either a database or a local file.")
    parser.add_argument("--database-url", help="Database URL for data retrieval.")
    parser.add_argument("--file", help="Path to the local file (CSV or Excel) to process.")

    args = parser.parse_args()

    if args.file:
        job(file_path=args.file)
    elif args.database_url:
        job(database_url=args.database_url)
    else:
        # Run immediately for testing or run in scheduled mode
        logging.info("Running job immediately as no args were provided.")
        job(file_path=r"C:\Users\amans\OneDrive\Documents\moviesdata.xlsx")
        # Comment out below for instant run, or uncomment for scheduling:
        # schedule.every().day.at("01:45").do(job, database_url='postgresql://username:password@localhost:5432/mydatabase')
        # while True:
        #     schedule.run_pending()
        #     time.sleep(1)