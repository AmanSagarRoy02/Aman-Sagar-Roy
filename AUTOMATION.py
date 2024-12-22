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
DOCUMENTS_DIR = os.path.abspath(os.path.expanduser(r"C:\Users\amans\OneDrive\Documents"))  # Replace with your path
DATA_AUTOMATION_DIR = os.path.join(DOCUMENTS_DIR, "data_automation")
PROCESSED_DATA_DIR = os.path.join(DATA_AUTOMATION_DIR, "processed_data")
VISUALIZATIONS_DIR = os.path.join(DATA_AUTOMATION_DIR, "visualizations")
LOG_FILE_PATH = os.path.join(DATA_AUTOMATION_DIR, "data_automation.log")

SCHEDULE_TYPE = "daily"
SCHEDULE_TIME = "00:00"
DATASET_PATH = None
DB_CONNECTION_STRING = None  # Set if using a database

# Logging setup
try:
    os.makedirs(DATA_AUTOMATION_DIR, exist_ok=True)
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
except OSError as e:
    print(f"Error creating directory or logging: {e}")
    logging.critical(f"Error creating directory or logging: {e}")
    exit(1)

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_data(data_source, file_format=None):
    """Loads data from various file formats or a database."""
    print(f"Loading data from: {data_source}")
    logging.info(f"Loading data from: {data_source}")
    df = None
    try:
        if isinstance(data_source, str) and os.path.exists(data_source):
            if not file_format:
                file_extension = os.path.splitext(data_source)[1].lower()
                if file_extension in [".csv", ".xlsx", ".xls", ".json"]:
                    file_format = file_extension.replace(".", "")
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")

            if file_format == "csv":
                df = pd.read_csv(data_source, low_memory=False)
            elif file_format in ["xlsx", "xls"]:
                df = pd.read_excel(data_source)
            elif file_format == "json":
                df = pd.read_json(data_source)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        elif DB_CONNECTION_STRING and isinstance(data_source, str):
            try:
                engine = sqlalchemy.create_engine(DB_CONNECTION_STRING)
                df = pd.read_sql_table(data_source, engine)
            except ImportError:
                raise ImportError("SQLAlchemy or database driver not installed.")
            except sqlalchemy.exc.OperationalError as e:
                raise ConnectionError(f"Database connection error: {e}")
        else:
            raise ValueError("Invalid data source.")

        print("Data loaded successfully.")
        return df
    except (FileNotFoundError, ValueError, ImportError, ConnectionError) as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        logging.exception(f"An unexpected error occurred during data loading: {e}")
        return None

def process_data(df):
    """Processes data (handles various data types more robustly)."""
    print("Processing Data...")
    logging.info("Processing data...")
    if df is None:
        print("No data to process.")
        logging.info("No data to process.")
        return None

    try:
        if df.empty:
            print("DataFrame is empty. No processing needed.")
            return df

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).fillna(df[col].mode()[0] if not df[col].empty else "Missing")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col]).fillna(df[col].mode()[0] if not df[col].empty else pd.Timestamp.min)
            else:
                df[col] = df[col].astype(str).fillna("Missing")

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

    if df is None or df.empty:
        print("No data to visualize or DataFrame is empty.")
        logging.info("No data to visualize or DataFrame is empty.")
        return

    try:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(exclude=['number']).columns

        if num_cols.empty and cat_cols.empty:
            print("No numeric or categorical columns to visualize.")
            logging.info("No numeric or categorical columns to visualize.")
            return

        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col])
            plt.title(f"Distribution of {col}")
            filepath = os.path.join(VISUALIZATIONS_DIR, f"{col}_hist_{now}.png")
            plt.savefig(filepath)
            plt.close()

        for col in cat_cols:
            if 1 <= len(df[col].unique()) <= 20:  # Limit for readability
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

def automate_data_tasks(data_source, file_format=None):
    """Automates the entire process."""
    print("Starting Data Automation Task")
    df = load_data(data_source, file_format)
    if df is not None:
        df = process_data(df)
        if df is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"processed_data_{now}.csv")
            df.to_csv(processed_file_path, index=False)
            print(f"Processed data saved to: {processed_file_path}")
            logging.info(f"Processed data saved to: {processed_file_path}")
            visualize_data(df)

def schedule_tasks(data_source, schedule_type, schedule_time, file_format=None):
    """Schedules tasks."""
    try:
        if schedule_type == "daily":
            schedule.every().day.at(schedule_time).do(automate_data_tasks, data_source, file_format)
        elif schedule_type == "weekly":
            schedule.every().week.on(0).at(schedule_time).do(automate_data_tasks, data_source, file_format)  # 0 is Monday
        elif schedule_type == "monthly":
            schedule.every(30).days.at(schedule_time).do(automate_data_tasks, data_source, file_format) #Approx monthly
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
        DATASET_PATH = input("Enter dataset file path: ").strip().strip('"')
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"File not found: {DATASET_PATH}")
        file_format = input("Enter file format (csv, xlsx, xls, json or press enter for auto-detect): ").lower() or None

        SCHEDULE_TYPE = input(f"Enter schedule type (daily/weekly/monthly, or press Enter for default: {SCHEDULE_TYPE}): ").lower() or SCHEDULE_TYPE
        if SCHEDULE_TYPE not in ("daily", "weekly", "monthly"):
            raise ValueError("Invalid schedule type.")

        SCHEDULE_TIME = input(f"Enter schedule time (HH:MM, or press Enter for default: {SCHEDULE_TIME}): ") or SCHEDULE_TIME
        datetime.strptime(SCHEDULE_TIME, '%H:%M')

        automate_data_tasks(DATASET_PATH, file_format)  # Run once immediately
        schedule_tasks(DATASET_PATH, SCHEDULE_TYPE, SCHEDULE_TIME, file_format)

    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        logging.error(str(e))
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        logging.exception(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    main()