import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import schedule
import time
import os
from datetime import datetime
import logging
import sqlalchemy

# Configuration
DOCUMENTS_DIR = os.path.abspath(os.path.expanduser(r"C:\Users\amans\OneDrive\Documents"))
DATA_AUTOMATION_DIR = os.path.join(DOCUMENTS_DIR, "data_automation")
PROCESSED_DATA_DIR = os.path.join(DATA_AUTOMATION_DIR, "processed_data")
VISUALIZATIONS_DIR = os.path.join(DATA_AUTOMATION_DIR, "visualizations")
LOG_FILE_PATH = os.path.join(DATA_AUTOMATION_DIR, "data_automation.log")

SCHEDULE_TYPE = "daily"
SCHEDULE_TIME = "00:00"
DATASET_PATH = r"C:\Users\amans\OneDrive\Documents\moviesdata.xlsx"  # Default path
DB_CONNECTION_STRING = None

# Logging setup
try:
    os.makedirs(DATA_AUTOMATION_DIR, exist_ok=True)  # Create 'data_automation' directory if not exists
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
except OSError as e:
    print(f"Error creating directory or logging: {e}")
    logging.critical(f"Error creating directory or logging: {e}")
    exit(1)  # Exit the script if directory creation or logging setup fails

# Ensure subdirectories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Ensure 'processed_data' subdirectory is created
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_data(data_source):
    """Loads data from a file or database."""
    print(f"Loading data from: {data_source}")
    logging.info(f"Loading data from: {data_source}")
    df = None
    try:
        if isinstance(data_source, str) and os.path.exists(data_source):
            file_extension = os.path.splitext(data_source)[1].lower()
            if file_extension == ".csv":
                df = pd.read_csv(data_source, low_memory=False)
            elif file_extension == ".xlsx" or file_extension == ".xls":
                df = pd.read_excel(data_source)
            elif file_extension == ".json":
                df = pd.read_json(data_source)
            else:
                logging.error(f"Unsupported file format: {file_extension}")
                print(f"Unsupported file format: {file_extension}")
                return None
        elif DB_CONNECTION_STRING and isinstance(data_source, str):
            try:
                engine = sqlalchemy.create_engine(DB_CONNECTION_STRING)
                df = pd.read_sql_table(data_source, engine)
            except ImportError:
                logging.error("SQLAlchemy or the database driver is not installed.")
                print("SQLAlchemy or the database driver is not installed.")
                return None
            except sqlalchemy.exc.OperationalError as e:
                logging.error(f"Database connection error: {e}")
                print(f"Database connection error: {e}")
                return None
        else:
            logging.error("Invalid or non-existent data source provided.")
            print("Invalid or non-existent data source provided.")
            return None
        print("File/Data from database Loaded Successfully")
        return df
    except FileNotFoundError:
        print(f"Error Loading Data: File Not Found at {data_source}")
        logging.exception(f"Error loading data: File not found {data_source}")
        return None
    except Exception as e:
        print(f"Error Loading Data: {e}")
        logging.exception(f"Error loading data: {e}")
        return None

def process_data(df):
    """Processes data."""
    print("Processing Data...")
    logging.info("Processing data...")
    if df is None:
        print("No data to process.")
        logging.info("No data to process.")
        return None

    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df.loc[:, col] = df[col].fillna(df[col].mean())

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                df[col] = df[col].astype(str)
                df.loc[:, col] = df[col].fillna(df[col].mode()[0])
            except (TypeError, IndexError):
                df.loc[:, col] = df[col].fillna("Missing")
        print("Data Processing Complete")
        return df
    except Exception as e:
        print(f"Error During Data Processing: {e}")
        logging.exception(f"Error during data processing: {e}")
        return None

def visualize_data(df):
    """Generates visualizations."""
    print("Generating Visualizations...")
    logging.info("Generating visualizations...")
    if df is None:
        print("No data to visualize.")
        logging.info("No data to visualize.")
        return

    try:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_cols = df.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            for col in num_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col])
                plt.title(f"Distribution of {col}")
                filepath = os.path.join(VISUALIZATIONS_DIR, f"{col}_hist_{now}.png")
                plt.savefig(filepath)
                plt.close()

        cat_cols = df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            if 1 <= len(df[col].unique()) <= 20:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=col, data=df)
                plt.xticks(rotation=45, ha='right')
                plt.subplots_adjust(bottom=0.2)
                plt.title(f"Count of {col}")
                filepath = os.path.join(VISUALIZATIONS_DIR, f"{col}_count_{now}.png")
                plt.savefig(filepath)
                plt.close()

        if len(num_cols) > 1:
            corr_matrix = df.corr(numeric_only=True)
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            filepath = os.path.join(VISUALIZATIONS_DIR, f"corr_heatmap_{now}.png")
            plt.savefig(filepath)
            plt.close()
        else:
            logging.warning("Not enough numeric columns to generate correlation heatmap.")
            print("Not enough numeric columns to generate correlation heatmap.")

        print("Visualizations Saved")
    except Exception as e:
        print(f"Error During Visualization: {e}")
        logging.error(f"Error during visualization: {e}")

def automate_data_tasks(data_source):
    """Automates the entire process."""
    print("Starting Data Automation Task")
    df = load_data(data_source)
    if df is not None:
        df = process_data(df)
        if df is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"processed_data_{now}.csv")
            df.to_csv(processed_file_path, index=False)
            print(f"Processed data saved to: {processed_file_path}")
            logging.info(f"Processed data saved to: {processed_file_path}")
            visualize_data(df)

def schedule_tasks(data_source, schedule_type, schedule_time):
    """Schedules tasks."""
    try:
        if schedule_type == "daily":
            schedule.every().day.at(schedule_time).do(automate_data_tasks, data_source)
        elif schedule_type == "weekly":
            schedule.every().week.on(0).at(schedule_time).do(automate_data_tasks, data_source)
        elif schedule_type == "monthly":
            schedule.every(30).days.at(schedule_time).do(automate_data_tasks, data_source)
        else:
            raise ValueError("Invalid schedule type.")

        print(f"Tasks scheduled to run {schedule_type} at {schedule_time}.")
        logging.info(f"Tasks scheduled to run {schedule_type} at {schedule_time}.")

        while True:
            schedule.run_pending()
            time.sleep(1)

    except ValueError as e:
        print(str(e))
        logging.error(str(e))
    except Exception as e:
        print(f"Scheduling Error: {e}")
        logging.exception(f"Scheduling error: {e}")

def main():
    """Main function."""
    global DATASET_PATH, SCHEDULE_TYPE, SCHEDULE_TIME
    try:
        DATASET_PATH = input(f"Enter dataset file path (or press Enter for default: {DATASET_PATH}): ") or DATASET_PATH
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"File not found: {DATASET_PATH}")

        SCHEDULE_TYPE = input(f"Enter schedule type (daily/weekly/monthly, or press Enter for default: {SCHEDULE_TYPE}): ").lower() or SCHEDULE_TYPE
        if SCHEDULE_TYPE not in ("daily", "weekly", "monthly"):
            raise ValueError("Invalid schedule type.")

        SCHEDULE_TIME = input(f"Enter schedule time (HH:MM, or press Enter for default: {SCHEDULE_TIME}): ") or SCHEDULE_TIME
        datetime.strptime(SCHEDULE_TIME, '%H:%M')

        automate_data_tasks(DATASET_PATH)  # Run once immediately
        schedule_tasks(DATASET_PATH, SCHEDULE_TYPE, SCHEDULE_TIME)

    except FileNotFoundError as e:
        print(str(e))
        logging.error(str(e))
    except ValueError as e:
        print(str(e))
        logging.error(str(e))
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        logging.exception(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    main()