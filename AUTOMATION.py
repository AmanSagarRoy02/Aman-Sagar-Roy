import os
import pandas as pd
import sqlite3
import traceback  # Import traceback for detailed error logging
import matplotlib
import numpy as np
matplotlib.use('agg')  # For non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import dash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix)
import warnings

try:
    from sklearn.inspection import PartialDependenceDisplay
except ImportError:
    print("Warning: PartialDependenceDisplay not found. PDP plots will be unavailable. Check scikit-learn version.")
    PartialDependenceDisplay = None  # Define as None if import fails

import plotly
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table, Input, Output
import json
import schedule
import time
import datetime
import signal
import sys
import logging
from sklearn.tree import plot_tree
import base64
import threading
import datetime

try:
    import shap

    shap_available = True
except ImportError:
    print("Warning: SHAP library not installed. SHAP analysis will be unavailable. Run `pip install shap`")
    shap_available = False
    shap = None  # Define as None if import fails

from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_object_dtype, \
    is_integer_dtype
from pandas import CategoricalDtype

# Configure logging
# Remove existing handlers before adding new ones to prevent duplicate logs
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,  # Set to DEBUG for more detailed info during debugging
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode='w'  # Overwrite log file each run for cleaner debugging
)
# Also log to console
console_handler_root = logging.StreamHandler(sys.stdout)
console_handler_root.setLevel(logging.INFO)  # Or DEBUG
console_handler_root.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
root_logger.addHandler(console_handler_root)

logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger


# LoggerFactory class (Simplified as we configure root logger directly now)
class LoggerFactory:
    @staticmethod
    def get_logger(name):
        # Simply return a logger instance; configuration is done globally
        return logging.getLogger(name)


# DataScienceAutomation Class
class DataScienceAutomation:
    def __init__(self):
        # Use LoggerFactory to get named loggers if needed, but base config is done
        self.pipeline_logger = LoggerFactory.get_logger("pipeline")
        self.scheduler_logger = LoggerFactory.get_logger("scheduler")

        # Data and model-related members
        self.data = None
        self.original_data = None
        self.processed_data = None
        self.target_col = None
        self.model_rf = None
        self.model_knn = None
        self.predictions_rf_test = None
        self.predictions_knn_test = None
        self.y_test = None
        self.X_test_original = None  # Store original X_test for reference if needed
        self.X_test_processed = None  # Store processed test features for SHAP/PDP
        self.clusters = None
        self.shap_values = None
        self.label_encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_names = []  # Initialize as empty list
        self.task_type = None  # 'classification' or 'regression'
        self.visualization_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.cluster_analysis_df = None
        # self.scaler = StandardScaler() # Scaler is part of the pipeline now

        # --- NEW: Data source parameters for automatic reloading ---
        self.data_source_type = None  # 'file' or 'database'
        self.file_path = None  # Path to the data file
        self.file_type = None  # Extension of the file (e.g., '.csv')
        self.file_delimiter = None  # For text files
        self.db_connection_string = None  # For database connections (path for sqlite, string for sqlalchemy)
        self.db_query = None  # SQL query for database loading
        self.db_type = None  # 'sqlite' or 'sqlalchemy'
        # --- End NEW ---

        self.pipeline_logger.info("DataScienceAutomation initialized")
        self.dashboard_running = False  # Flag to track dashboard status

    def graceful_exit(self, signal_received, frame):
        self.pipeline_logger.info("Shutdown signal received. Stopping gracefully.")
        # Add any specific cleanup here if needed
        sys.exit(0)

    # =================== Data Ingestion Methods ===================

    def prompt_for_dataset(self):
        print("\n--- Data Source Selection ---")
        print("1. Local File (CSV, Excel, JSON, Parquet, TXT)")
        print("2. Database (SQLite, SQLAlchemy)")
        while True:
            choice = input("Enter choice (1/2): ").strip()
            if choice == '1':
                self.load_file()
                break
            elif choice == '2':
                self.load_database()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def load_file(self):
        while True:
            try:
                file_path_input = input("Enter the full path to your dataset file: ").strip()
                # Remove surrounding quotes if present (e.g., from drag-and-drop)
                file_path = file_path_input.strip('"').strip("'")

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found at the specified path: {file_path}")

                ext = os.path.splitext(file_path)[1].lower()
                self.pipeline_logger.info(f"Attempting to load file: {file_path} (type: {ext})")
                delimiter = None  # Initialize delimiter for this scope

                if ext == '.csv':
                    # Try common encodings
                    try:
                        self.data = pd.read_csv(file_path)
                    except UnicodeDecodeError:
                        self.pipeline_logger.warning("UTF-8 decoding failed, trying ISO-8859-1...")
                        self.data = pd.read_csv(file_path, encoding='ISO-8859-1')
                    except Exception as e:
                        self.pipeline_logger.error(f"Failed to read CSV with multiple encodings: {e}")
                        raise
                elif ext in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(file_path)
                elif ext == '.json':
                    self.data = pd.read_json(file_path)
                elif ext == '.parquet':
                    self.data = pd.read_parquet(file_path)
                elif ext == '.txt':
                    delimiter = input("Enter the delimiter for the text file (e.g., ',' or '\\t'): ").strip()
                    if not delimiter:
                        delimiter = ','  # Default to comma if empty
                        print("Defaulting to comma delimiter.")
                    try:
                        self.data = pd.read_csv(file_path, delimiter=delimiter)
                    except UnicodeDecodeError:
                        self.pipeline_logger.warning("UTF-8 decoding failed for TXT, trying ISO-8859-1...")
                        self.data = pd.read_csv(file_path, delimiter=delimiter, encoding='ISO-8859-1')
                    except Exception as e:
                        self.pipeline_logger.error(f"Failed to read TXT file: {e}")
                        raise
                else:
                    raise ValueError(
                        f"Unsupported file format '{ext}'. Supported formats: .csv, .xlsx, .xls, .json, .parquet, .txt")

                if self.data is None or self.data.empty:
                    raise ValueError("Loaded data is empty. Please check the file content and format.")

                self.pipeline_logger.info(
                    f"Successfully loaded data from {file_path}. Initial shape: {self.data.shape}")

                # --- Store parameters for automatic reload ---
                self.data_source_type = 'file'
                self.file_path = file_path
                self.file_type = ext
                self.file_delimiter = delimiter if ext == '.txt' else None  # Store delimiter only if txt
                self.db_connection_string = None  # Clear DB params if loading file
                self.db_query = None
                self.db_type = None
                self.pipeline_logger.info(
                    f"Stored file source parameters for future reloads: path={self.file_path}, type={self.file_type}")
                # --- End Store parameters ---

                # --- Initial Data Cleaning ---
                # Drop columns with >80% missing values
                initial_cols = self.data.shape[1]
                missing_threshold = 0.8
                high_missing_cols = self.data.columns[self.data.isnull().mean() > missing_threshold]
                if not high_missing_cols.empty:
                    self.pipeline_logger.warning(
                        f"Dropping columns with >{missing_threshold * 100:.1f}% missing values: {list(high_missing_cols)}")
                    self.data.drop(columns=high_missing_cols, inplace=True)
                    self.pipeline_logger.info(f"Shape after dropping high missing columns: {self.data.shape}")

                # Drop rows that are completely empty
                initial_rows = self.data.shape[0]
                self.data.dropna(axis=0, how='all', inplace=True)
                if self.data.shape[0] < initial_rows:
                    self.pipeline_logger.info(
                        f"Dropped {initial_rows - self.data.shape[0]} fully empty rows. Shape now: {self.data.shape}")

                if self.data.empty:
                    raise ValueError("Data became empty after initial cleaning (dropping missing columns/rows).")

                self.original_data = self.data.copy()  # Store the cleaned version as original for reprocessing
                print("\n--- Data Loaded Successfully ---")
                print(f"Shape: {self.data.shape}")
                print(f"Columns: {list(self.data.columns)}")
                print("\nData Sample (first 5 rows):")
                print(self.data.head())
                break  # Exit loop if successful

            except FileNotFoundError as e:
                self.pipeline_logger.error(f"Data loading error: {e}")
                print(f"\nError: {e}. Please check the path and try again.")
            except ValueError as e:
                self.pipeline_logger.error(f"Data loading/validation error: {e}")
                print(f"\nError: {e}. Please check the file content/format and try again.")
            except Exception as e:
                self.pipeline_logger.error(f"An unexpected error occurred during file loading: {e}", exc_info=True)
                print(f"\nAn unexpected error occurred: {e}. Check pipeline.log for details.")
                # Optionally re-raise or exit if critical
                raise  # Re-raise unexpected errors

    def load_database(self):
        try:
            db_type = input("Enter the database type ('sqlite' or 'sqlalchemy'): ").strip().lower()
            db_path = None  # Initialize
            conn_string = None  # Initialize
            if db_type == 'sqlite':
                db_path = input("SQLite File Path: ").strip().strip('"').strip("'")
                conn = sqlite3.connect(db_path)
                query = input("SQL Query: ").strip()
                self.data = pd.read_sql_query(query, conn)
                conn.close()
            elif db_type == 'sqlalchemy':
                conn_string = input("SQLAlchemy connection string: ").strip()
                engine = create_engine(conn_string)
                query = input("SQL Query: ").strip()
                self.data = pd.read_sql_query(query, engine)
                # Dispose engine? Maybe not necessary for read-only?
                # engine.dispose()
            else:
                raise ValueError("Unsupported database type. Please enter 'sqlite' or 'sqlalchemy'.")

            if self.data is None or self.data.empty:
                raise ValueError("Loaded DataFrame from database is empty or None.")

            # --- Store parameters for automatic reload ---
            self.data_source_type = 'database'
            self.db_connection_string = db_path if db_type == 'sqlite' else conn_string  # Store relevant connection info
            self.db_query = query
            self.db_type = db_type  # Store 'sqlite' or 'sqlalchemy'
            self.file_path = None  # Clear file params if loading DB
            self.file_type = None
            self.file_delimiter = None
            self.pipeline_logger.info(f"Stored database source parameters for future reloads: type={self.db_type}")
            # --- End Store parameters ---

            # --- Initial Cleaning (Same as load_file) ---
            initial_cols = self.data.shape[1]
            missing_threshold = 0.8
            high_missing_cols = self.data.columns[self.data.isnull().mean() > missing_threshold]
            if not high_missing_cols.empty:
                self.pipeline_logger.warning(
                    f"Dropping columns with >{missing_threshold * 100:.1f}% missing values: {list(high_missing_cols)}")
                self.data.drop(columns=high_missing_cols, inplace=True)

            initial_rows = self.data.shape[0]
            self.data.dropna(axis=0, how='all', inplace=True)
            if self.data.shape[0] < initial_rows:
                self.pipeline_logger.info(
                    f"Dropped {initial_rows - self.data.shape[0]} fully empty rows.")

            if self.data.empty:
                raise ValueError("Data became empty after initial cleaning.")

            self.original_data = self.data.copy()
            self.pipeline_logger.info(f"Dataset loaded from database; shape: {self.data.shape}")
            print("\n--- Data Loaded Successfully ---")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            print("\nData Sample (first 5 rows):")
            print(self.data.head())
        except Exception as e:
            self.pipeline_logger.error(f"Error loading database data: {e}", exc_info=True)
            print(f"\nAn error occurred loading from database: {e}. Check logs.")
            raise

    def _reload_data_automatically(self):
        """ Loads data non-interactively using stored parameters. """
        self.pipeline_logger.info(f"--- Starting Automatic Data Reload (Source Type: {self.data_source_type}) ---")
        if self.data_source_type == 'file' and self.file_path:
            try:
                file_path = self.file_path
                ext = self.file_type
                delimiter = self.file_delimiter
                self.pipeline_logger.info(f"Attempting to auto-reload file: {file_path} (type: {ext})")

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Automatic reload failed: File not found at stored path: {file_path}")

                if ext == '.csv':
                    try:
                        self.data = pd.read_csv(file_path)
                    except UnicodeDecodeError:
                        self.pipeline_logger.warning("UTF-8 decoding failed during auto-reload, trying ISO-8859-1...")
                        self.data = pd.read_csv(file_path, encoding='ISO-8859-1')
                elif ext in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(file_path)
                elif ext == '.json':
                    self.data = pd.read_json(file_path)
                elif ext == '.parquet':
                    self.data = pd.read_parquet(file_path)
                elif ext == '.txt':
                    if delimiter is None:  # Should have been stored, but default if missing
                        self.pipeline_logger.warning("Delimiter not stored for TXT auto-reload, defaulting to ','.")
                        delimiter = ','
                    try:
                        self.data = pd.read_csv(file_path, delimiter=delimiter)
                    except UnicodeDecodeError:
                        self.pipeline_logger.warning("UTF-8 decoding failed for TXT auto-reload, trying ISO-8859-1...")
                        self.data = pd.read_csv(file_path, delimiter=delimiter, encoding='ISO-8859-1')
                else:
                    # This shouldn't happen if parameters were stored correctly
                    raise ValueError(f"Unsupported file type '{ext}' encountered during auto-reload.")

                if self.data is None or self.data.empty:
                    raise ValueError("Auto-reloaded data is empty. Check the source file.")

                self.pipeline_logger.info(f"Successfully auto-reloaded data from {file_path}. Shape: {self.data.shape}")

            except Exception as e:
                self.pipeline_logger.error(f"Automatic file reload failed: {e}", exc_info=True)
                raise  # Re-raise to stop the scheduled job

        elif self.data_source_type == 'database' and self.db_connection_string and self.db_query:
            try:
                db_type = self.db_type
                conn_string_or_path = self.db_connection_string
                query = self.db_query
                self.pipeline_logger.info(f"Attempting to auto-reload from database (type: {db_type})")

                if db_type == 'sqlite':
                    conn = sqlite3.connect(conn_string_or_path)
                    self.data = pd.read_sql_query(query, conn)
                    conn.close()
                elif db_type == 'sqlalchemy':
                    engine = create_engine(conn_string_or_path)
                    self.data = pd.read_sql_query(query, engine)
                    # Dispose engine? Maybe not necessary for read-only?
                    # engine.dispose()
                else:
                    raise ValueError(f"Unsupported database type '{db_type}' encountered during auto-reload.")

                if self.data is None or self.data.empty:
                    raise ValueError("Auto-reloaded data from database is empty.")

                self.pipeline_logger.info(f"Successfully auto-reloaded data from database. Shape: {self.data.shape}")

            except Exception as e:
                self.pipeline_logger.error(f"Automatic database reload failed: {e}", exc_info=True)
                raise  # Re-raise to stop the scheduled job
        else:
            msg = "Cannot reload data automatically: Data source parameters not set or invalid."
            self.pipeline_logger.error(msg)
            raise ValueError(msg)  # Stop the scheduled job

        # --- Apply Initial Cleaning (same as in load_file) ---
        # !!! CORRECTED INDENTATION STARTS HERE !!!
        try:
            initial_cols = self.data.shape[1]
            missing_threshold = 0.8
            high_missing_cols = self.data.columns[self.data.isnull().mean() > missing_threshold]
            if not high_missing_cols.empty:
                self.pipeline_logger.warning(
                    f"(Auto-reload) Dropping columns with >{missing_threshold * 100:.1f}% missing values: {list(high_missing_cols)}")
                self.data.drop(columns=high_missing_cols, inplace=True)

            initial_rows = self.data.shape[0]
            self.data.dropna(axis=0, how='all', inplace=True)
            if self.data.shape[0] < initial_rows:
                self.pipeline_logger.info(
                    f"(Auto-reload) Dropped {initial_rows - self.data.shape[0]} fully empty rows.")

            if self.data.empty:
                raise ValueError("Data became empty after initial cleaning during auto-reload.")

            self.original_data = self.data.copy()  # IMPORTANT: Update original_data after successful auto-reload
            self.pipeline_logger.info(f"Automatic data reload and initial cleaning complete. Shape: {self.data.shape}")

        except Exception as clean_err:
            self.pipeline_logger.error(f"Error during initial cleaning after auto-reload: {clean_err}", exc_info=True)
            raise  # Re-raise cleaning errors
        # !!! CORRECTED INDENTATION ENDS HERE !!!

    # =================== Utility / Internal Methods ===================

    def _detect_task_type(self, y):
        """Detects task type (classification/regression) based on target column dtype and uniqueness."""
        if y is None or len(y) == 0:
            raise ValueError("Target column 'y' is empty or None, cannot detect task type.")
        # Ensure y is a pandas Series for nunique()
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        n_unique = y.nunique()
        length = len(y)
        unique_ratio = n_unique / length if length > 0 else 0

        self.pipeline_logger.info(
            f"Detecting task type for target '{y.name}': dtype={y.dtype}, unique={n_unique}, ratio={unique_ratio:.4f}")

        # Explicitly check for boolean type -> classification
        if pd.api.types.is_bool_dtype(y):
            self.pipeline_logger.info(f"Detected boolean target '{y.name}' as classification.")
            return 'classification'

        # Check numeric types
        if pd.api.types.is_numeric_dtype(y):
            # Binary classification (0/1 or two distinct numbers)
            if n_unique == 2:
                self.pipeline_logger.info(
                    f"Detected numeric target '{y.name}' with 2 unique values as classification.")
                return 'classification'
            # Low cardinality integer/float -> likely classification (heuristic)
            # Increased threshold slightly, maybe < 5% unique and < 100 unique values?
            if unique_ratio < 0.05 and n_unique < 100:
                # Further check: if all values are integers, more likely classification
                # Check for NaN before attempting conversion to int
                y_dropna = y.dropna()
                if not y_dropna.empty and (
                        pd.api.types.is_integer_dtype(y) or (y_dropna == y_dropna.astype(int)).all()):
                    self.pipeline_logger.info(
                        f"Detected numeric target '{y.name}' with low cardinality ({n_unique} unique, int-like) as classification.")
                    return 'classification'

            # Otherwise, assume regression for numeric types
            self.pipeline_logger.info(f"Detected numeric target '{y.name}' as regression.")
            return 'regression'
        else:
            # Non-numeric (object, category) is classification
            self.pipeline_logger.info(f"Detected non-numeric target '{y.name}' as classification.")
            return 'classification'

    def _apply_clustering_on_features(self, numeric_features_df):
        """ Helper to apply clustering specifically on numeric feature data (X). """
        if numeric_features_df is None or numeric_features_df.empty:
            self.pipeline_logger.warning("No numeric features provided for clustering. Skipping feature clustering.")
            self.cluster_analysis_df = None
            return numeric_features_df  # Return the original df

        self.pipeline_logger.info(f"Applying clustering on {numeric_features_df.shape[1]} numeric features...")
        clustered_df = numeric_features_df.copy()  # Work on a copy

        try:
            # Data should already be imputed before calling this
            if clustered_df.isnull().any().any():
                self.pipeline_logger.warning(
                    "Missing values found in numeric features passed to clustering. Imputing with median.")
                num_imputer = SimpleImputer(strategy='median')
                numeric_features_imputed = num_imputer.fit_transform(clustered_df)
                clustered_df = pd.DataFrame(numeric_features_imputed, columns=clustered_df.columns,
                                            index=clustered_df.index)

            # Scale data before clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(clustered_df)

            # Determine optimal K (using 3 as default, ensure K is reasonable)
            # Ensure at least 2 points per cluster, and max K reasonable (e.g., 10)
            n_clusters = min(max(2, len(clustered_df) // 10), 10)  # Heuristic: 2 to 10 clusters
            if len(clustered_df) < n_clusters * 2:  # Check if enough data points
                n_clusters = max(2, len(clustered_df) // 2)  # Adjust if too few points

            if n_clusters < 2:
                self.pipeline_logger.warning(
                    f"Not enough data points ({len(clustered_df)}) for meaningful clustering (k={n_clusters}). Skipping feature clustering.")
                self.cluster_analysis_df = None
                return numeric_features_df  # Return original df

            self.pipeline_logger.info(f"Using k={n_clusters} for KMeans clustering.")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # Set n_init explicitly
            cluster_labels = kmeans.fit_predict(features_scaled)

            # Add cluster labels back to the DataFrame copy
            clustered_df['cluster'] = cluster_labels
            self.pipeline_logger.info(f"Added 'cluster' column with values ranging from 0 to {cluster_labels.max()}.")

            # Calculate cluster profiles based on the numeric features used for clustering
            self.cluster_analysis_df = clustered_df.groupby('cluster').mean()  # numeric_only=True is default

            cluster_profile_path = os.path.join(self.visualization_dir, "cluster_profiles.csv")
            if self.cluster_analysis_df is not None:
                try:
                    self.cluster_analysis_df.to_csv(cluster_profile_path)
                    self.pipeline_logger.info(f"Cluster profiles (based on features) saved to {cluster_profile_path}")
                except Exception as write_err:
                    self.pipeline_logger.error(f"Failed to save cluster profiles: {write_err}")
            else:
                self.pipeline_logger.warning("Cluster analysis DataFrame is None, skipping saving profiles.")

            return clustered_df  # Return the dataframe with the 'cluster' column added

        except Exception as e:
            self.pipeline_logger.error(f"Feature clustering failed: {str(e)}", exc_info=True)
            self.cluster_analysis_df = None
            # Ensure 'cluster' column doesn't exist if clustering failed mid-way
            if 'cluster' in clustered_df.columns:
                del clustered_df['cluster']
            return numeric_features_df  # Return the original df on error

    def _parse_json_safe(self, value):
        # Improved JSON parsing
        if isinstance(value, str):
            value_stripped = value.strip()
            if value_stripped.startswith("{") and value_stripped.endswith("}"):
                try:
                    # More robust parsing: handle potential single quotes, common errors
                    formatted_value = value.replace("'", '"')
                    # Basic cleanup: remove trailing comma before closing brace/bracket
                    if formatted_value.endswith(",}"): formatted_value = formatted_value[:-2] + "}"
                    if formatted_value.endswith(",]"): formatted_value = formatted_value[:-2] + "]"

                    parsed = json.loads(formatted_value)
                    # Extract 'name' if it exists, otherwise return the whole dict? Or a placeholder?
                    # Returning the dict might break downstream if expecting scalar. Let's try 'name' or first value.
                    if isinstance(parsed, dict):
                        return parsed.get('name',
                                          next(iter(parsed.values()), value_stripped) if parsed else value_stripped)
                    else:
                        return parsed  # If it parsed but wasn't a dict
                except (json.JSONDecodeError, TypeError, StopIteration):
                    self.pipeline_logger.debug(f"Could not parse value as JSON object: {value}")
                    return value_stripped  # Return original stripped value if parsing fails
                except Exception as e:
                    self.pipeline_logger.warning(f"Unexpected error parsing JSON-like value '{value}': {e}")
                    return value_stripped  # Return original stripped value on other errors
        return value

    def _process_datetime_column(self, df, col):
        """Processes a datetime column in the given DataFrame (df)."""
        try:
            # Ensure the column is actually datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                self.pipeline_logger.info(f"Attempting conversion of '{col}' to datetime...")
                df[col] = pd.to_datetime(df[col], errors='coerce')

            # Drop if all values became NaT after conversion
            if df[col].isnull().all():
                self.pipeline_logger.warning(f"Column '{col}' resulted in all NaT after datetime conversion. Dropping.")
                return df.drop(columns=[col])  # Return modified df

            self.pipeline_logger.info(f"Extracting features from datetime column: {col}")
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            # Only add hour/minute if they have variance
            if df[col].dt.hour.nunique() > 1:
                df[f'{col}_hour'] = df[col].dt.hour
            if df[col].dt.minute.nunique() > 1:
                df[f'{col}_minute'] = df[col].dt.minute
            # Consider adding is_weekend, time_of_day etc.

            # Impute potential NaNs created during dt extraction (e.g., from NaT dates)
            new_dt_cols = [c for c in df.columns if c.startswith(f'{col}_')]
            # new_dt_cols = [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek', f'{col}_hour', f'{col}_minute']
            for new_col in new_dt_cols:
                if new_col in df.columns and df[new_col].isnull().any():
                    # Impute with median for numeric extracted features
                    impute_val = df[new_col].median()
                    # Fix FutureWarning by avoiding chained assignment with inplace=True
                    df[new_col] = df[new_col].fillna(impute_val)
                    self.pipeline_logger.info(
                        f"Imputed NaNs in extracted datetime feature '{new_col}' with median {impute_val}")

            # Drop the original datetime column
            df = df.drop(columns=[col])
            self.pipeline_logger.info(f"Dropped original datetime column '{col}'.")
            return df

        except Exception as e:
            self.pipeline_logger.error(f"Error processing datetime column '{col}': {e}", exc_info=True)
            # Decide if we should drop the original column even if processing fails
            if col in df.columns:
                self.pipeline_logger.warning(
                    f"Dropping original column '{col}' after error during datetime processing.")
                return df.drop(columns=[col])
            return df  # Return df even if error occurred

    # =================== Data Processing Methods ===================


    def process_data(self, manual_target_col=None, is_initial_run=False):  # Add is_initial_run here
        """
            Processes the loaded data: handles duplicates, JSON, datetime, missing values,
            target selection, encoding, and feature engineering (clustering).

            Args:
                manual_target_col (str, optional): Manually specified target column for this run.
                is_initial_run (bool): Flag indicating if this is the very first execution.
            """
        try:
            self.pipeline_logger.info("--- Starting Data Processing ---")
            if self.original_data is None:
                # This should ideally not happen if load methods are called first
                self.pipeline_logger.error("Original data is None. Attempting to load.")
                self.prompt_for_dataset()
                if self.original_data is None:
                    raise ValueError("Data could not be loaded.")

            # Start fresh from the original data for each run
            self.data = self.original_data.copy()
            self.pipeline_logger.info(f"Working with data shape: {self.data.shape}")

            # --- Basic Cleaning ---
            initial_rows = len(self.data)
            self.data.drop_duplicates(inplace=True)
            if len(self.data) < initial_rows:
                self.pipeline_logger.info(
                    f"Dropped {initial_rows - len(self.data)} duplicate rows. Shape now: {self.data.shape}")

            # Drop completely empty columns
            cols_to_drop = [col for col in self.data.columns if self.data[col].isnull().all()]
            if cols_to_drop:
                self.pipeline_logger.warning(f"Dropping completely empty columns: {cols_to_drop}")
                self.data.drop(columns=cols_to_drop, inplace=True)

            if self.data.empty:
                raise ValueError("Data is empty after removing duplicates and empty columns.")

            # --- Type-Specific Processing ---
            # 1. Parse JSON-like columns
            object_cols = self.data.select_dtypes(include='object').columns
            for col in object_cols:
                # Check if *any* non-null value looks like a JSON object string
                # Use .loc to avoid SettingWithCopyWarning if possible
                if self.data.loc[self.data[col].notna(), col].astype(str).str.strip().str.startswith('{').any():
                    self.pipeline_logger.info(f"Attempting parsing JSON-like data in column '{col}'")
                    self.data.loc[:, col] = self.data[col].apply(self._parse_json_safe)

            # 2. Handle Datetime Columns (existing and convertible objects)
            # Process existing datetime columns first
            datetime_cols = self.data.select_dtypes(
                include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns.tolist()
            self.pipeline_logger.info(f"Found existing datetime columns: {datetime_cols}")
            for col in datetime_cols:
                self.data = self._process_datetime_column(self.data, col)  # Pass and reassign df

            # Attempt to convert object columns to datetime
            # Re-check object columns after potential JSON parsing
            potential_dt_cols = self.data.select_dtypes(include='object').columns
            self.pipeline_logger.info(
                f"Checking potential datetime columns in object types: {potential_dt_cols.tolist()}")
            for col in potential_dt_cols:
                if col not in self.data.columns: continue  # Skip if already dropped
                # Heuristic: Try conversion if a sample looks like dates/times
                try:
                    sample = self.data[col].dropna().sample(min(50, len(self.data[col].dropna())))
                    if sample.empty: continue
                    # Suppress UserWarning during sample check
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        pd.to_datetime(sample, errors='raise')  # Try converting a sample

                    # If sample conversion works, try converting the whole column and process
                    self.pipeline_logger.info(f"Attempting to convert object column '{col}' to datetime.")
                    original_col_data = self.data[col].copy()  # Keep original in case conversion fails partially
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    # Check if conversion was successful for a significant portion
                    if not self.data[col].isnull().all() and pd.api.types.is_datetime64_any_dtype(self.data[col]):
                        self.data = self._process_datetime_column(self.data, col)  # Pass and reassign df
                    else:
                        self.pipeline_logger.warning(
                            f"Conversion of '{col}' to datetime failed or resulted in all NaT. Reverting to original.")
                        self.data[col] = original_col_data  # Revert if conversion wasn't useful

                except (ValueError, TypeError, OverflowError) as dt_err:
                    self.pipeline_logger.debug(f"Column '{col}' not convertible to datetime: {dt_err}")
                    continue  # Continue to next column if conversion fails

            self.pipeline_logger.info(f"Shape after datetime processing: {self.data.shape}")


            # --- Target Column Handling ---
            # 3. Identify Target Column
            try:
                target_to_use = None  # Initialize variable to hold the target for this run

                # --- Initial Run Logic ---
                if is_initial_run:
                    self.pipeline_logger.info("Initial run: Forcing interactive target column selection.")
                    print("\n--- Target Column Selection (Initial Run) ---")
                    print("This is the first run. Please select the target column.")
                    available_columns = self.data.columns.tolist()
                    for i, col in enumerate(available_columns, start=1):
                        print(f"{i}. {col}")

                    suggested_target = None
                    try:
                        suggested_target = self.select_target_column()
                        print(f"\nSuggested target based on analysis: '{suggested_target}'")
                        # Prompt message will be set inside the loop now
                    except ValueError as e:
                        self.pipeline_logger.warning(f"Auto-detection failed during initial prompt: {e}")
                        # Prompt message will be set inside the loop now

                    # --- Start of Replacement for Initial Run Prompt Loop ---
                    while True:
                        # Adjust prompt based on whether a suggestion exists
                        if suggested_target:
                            prompt_msg = f"Enter target column name (or press Enter for '{suggested_target}'): "
                        else:
                            prompt_msg = "Enter the exact target column name (required): "

                        user_input = input(prompt_msg).strip()

                        if not user_input:  # User pressed Enter
                            if suggested_target:
                                target_to_use = suggested_target
                                self.pipeline_logger.info(f"User accepted suggested target for initial run: {target_to_use}")
                                break  # Exit loop
                            else:
                                # No suggestion available, Enter is not valid
                                print("Error: Input is required as no suggestion could be made. Please enter a valid column name.")
                                # Loop continues
                        elif user_input in self.data.columns:  # User typed a valid column name
                            target_to_use = user_input
                            self.pipeline_logger.info(f"User selected target column for initial run: {target_to_use}")
                            break  # Exit loop
                        else:  # User typed something, but it's not a valid column
                            print(f"Error: Column '{user_input}' not found. Please choose from the list above or check spelling.")
                            # Loop continues
                    # --- End of Replacement for Initial Run Prompt Loop ---

                # --- Subsequent Run Logic ---
                else:  # Not the initial run
                    self.pipeline_logger.info("Subsequent run: Checking for stored target column.")
                    # Priority 1: Use manually provided target for this specific run (if any)
                    if manual_target_col and manual_target_col in self.data.columns:
                        target_to_use = manual_target_col
                        self.pipeline_logger.info(f"Using provided manual target column for this run: {target_to_use}")
                    # Priority 2: Use previously stored target if it's still valid
                    elif self.target_col and self.target_col in self.data.columns:
                        target_to_use = self.target_col
                        self.pipeline_logger.info(f"Using previously stored target column: {target_to_use}")
                    # Priority 3: Stored target is invalid or missing, proceed to auto-detect/prompt
                    else:
                        if self.target_col and self.target_col not in self.data.columns:
                            self.pipeline_logger.warning(
                                f"Previously stored target '{self.target_col}' not found in current data. Re-detecting.")
                        else:  # self.target_col was None or never set
                            self.pipeline_logger.info("No valid stored target found. Proceeding with auto-detection.")

                        # Auto-detect a suggestion
                        auto_target = self.select_target_column()  # May raise ValueError if fails
                        self.pipeline_logger.info(f"Auto-selected target column candidate: {auto_target}")

                        # Check if interactive for prompting
                        is_interactive = sys.stdin.isatty()
                        if is_interactive:
                            print(f"\n--- Target Column Selection ---")
                            print(f"Available columns: {list(self.data.columns)}")
                            if self.target_col:  # Inform user why we are prompting again
                                print(
                                    f"(Note: Previous target '{self.target_col}' was not found or invalid in the current data)")
                            print(f"Auto-selected target column is: '{auto_target}'")
                            confirm = input("Press Enter to accept, or type a different column name: ").strip()
                            if confirm and confirm in self.data.columns:
                                target_to_use = confirm
                                self.pipeline_logger.info(f"User selected target column: {target_to_use}")
                            elif confirm and confirm not in self.data.columns:
                                self.pipeline_logger.warning(
                                    f"User input '{confirm}' not a valid column. Using auto-selected '{auto_target}'.")
                                print(f"Warning: '{confirm}' not found. Using auto-selected target '{auto_target}'.")
                                target_to_use = auto_target
                            else:  # User pressed Enter or provided invalid input when suggestion exists
                                target_to_use = auto_target
                                self.pipeline_logger.info(f"User accepted auto-selected target column: {target_to_use}")
                        else:  # Non-interactive (e.g., scheduled run)
                            self.pipeline_logger.info(
                                f"Non-interactive mode detected. Using auto-selected target: {auto_target}")
                            target_to_use = auto_target

                # --- Final Assignment and Validation ---
                if target_to_use and target_to_use in self.data.columns:
                    self.target_col = target_to_use  # Store/update the class attribute for future runs
                    self.pipeline_logger.info(f"Final target column set to: {self.target_col}")
                else:
                    # This case should be rare now, especially after the enforced initial run prompt
                    self.pipeline_logger.error("Failed to determine a valid target column after all checks.")
                    # If non-interactive, raise error. If interactive, it implies a failure during the prompt.
                    if not sys.stdin.isatty():
                        raise ValueError("Could not determine a valid target column in non-interactive mode.")
                    else:
                        # Should have been caught by the initial run loop, but raise anyway if we get here
                        raise ValueError("Failed to get a valid target column during interactive selection.")

            except ValueError as e:
                # Handle errors during auto-detection or if prompt fails
                self.pipeline_logger.error(f"Target column selection failed: {e}", exc_info=True)
                # If it's the initial run, the error likely came from the prompt loop or select_target_column
                if is_initial_run:
                    print(f"\nERROR: Critical failure during initial target selection: {e}")
                    raise  # Critical failure for initial run
                # If subsequent run and interactive, maybe prompt again? Safer to raise.
                elif sys.stdin.isatty():
                    print(f"\nERROR: Failed to select target column: {e}")
                    raise  # Raise error even if interactive on subsequent runs
                else:  # Non-interactive subsequent run
                    raise ValueError(f"Target column selection failed in non-interactive mode: {e}")  # Re-raise

            self.pipeline_logger.info(
                f"Target column for processing: {self.target_col}")  # Log the final decision for this run

            # --- Rest of the process_data method continues below ---
            # 4. Handle Missing Target Values (Impute or Drop Rows)
            if self.data[self.target_col].isnull().any():
                missing_pct = self.data[self.target_col].isnull().mean() * 100
                self.pipeline_logger.warning(
                    f"Target column '{self.target_col}' has {missing_pct:.2f}% missing values.")
                # Simple strategy: Drop rows with missing target
                initial_rows = len(self.data)
                self.data.dropna(subset=[self.target_col], inplace=True)
                self.pipeline_logger.warning(
                    f"Dropped {initial_rows - len(self.data)} rows with missing target values.")
                if self.data.empty:
                    raise ValueError(
                        f"Dataset became empty after dropping rows with missing target ('{self.target_col}').")

            # 5. Encode Target Column (if categorical/object) before splitting
            target_dtype = self.data[self.target_col].dtype
            if pd.api.types.is_object_dtype(target_dtype) or isinstance(target_dtype, pd.CategoricalDtype):
                self.pipeline_logger.info(f"Label encoding target column '{self.target_col}'.")
                le = LabelEncoder()
                # Ensure no NaNs before encoding (should be handled above, but double-check)
                self.data[self.target_col] = self.data[self.target_col].fillna('__MISSING__')  # Temp fill just in case
                self.data[self.target_col] = le.fit_transform(self.data[self.target_col].astype(str))
                # Store the encoder and its classes
                self.label_encoders[self.target_col] = le
                class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                self.pipeline_logger.info(f"Target classes mapped: {class_mapping}")
                if '__MISSING__' in le.classes_:
                    self.pipeline_logger.warning(
                        "Missing values were present and encoded in the target. This might affect model interpretation.")

            # --- Feature Engineering / Final Processing ---
            # 6. Separate Features (X) and Target (y)
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]  # Keep y separate

            # 7. Identify Feature Types (Numeric, Categorical) from X
            self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
            self.categorical_cols = X.select_dtypes(
                include=['object', 'category', 'bool']).columns.tolist()  # Include bool as categorical for OHE

            self.pipeline_logger.info(
                f"Identified Numerical Features ({len(self.numerical_cols)}): {self.numerical_cols}")
            self.pipeline_logger.info(
                f"Identified Categorical Features ({len(self.categorical_cols)}): {self.categorical_cols}")

            # 8. Impute Missing Features in X (will also be handled by pipeline, but can be done here for clustering)
            # Numerical Imputation (using median)
            if self.numerical_cols and X[self.numerical_cols].isnull().any().any():
                self.pipeline_logger.info("Applying median imputation to numerical features in X (pre-pipeline)...")
                num_imputer = SimpleImputer(strategy='median')
                X.loc[:, self.numerical_cols] = num_imputer.fit_transform(X[self.numerical_cols])

            # Categorical Imputation (using constant '__MISSING__')
            if self.categorical_cols and X[self.categorical_cols].isnull().any().any():
                self.pipeline_logger.info(
                    "Applying constant ('__MISSING__') imputation to categorical features in X (pre-pipeline)...")
                cat_imputer = SimpleImputer(strategy='constant', fill_value='__MISSING__')
                X.loc[:, self.categorical_cols] = cat_imputer.fit_transform(X[self.categorical_cols])

            # 9. Apply Clustering (on imputed numeric features in X)
            if self.numerical_cols:
                X_clustered = self._apply_clustering_on_features(X[self.numerical_cols])
                # If clustering added a 'cluster' column, add it to X and update feature lists
                if 'cluster' in X_clustered.columns:
                    X['cluster'] = X_clustered['cluster']
                    # Treat cluster as a categorical feature
                    if 'cluster' not in self.categorical_cols:
                        self.categorical_cols.append('cluster')
                    if 'cluster' in self.numerical_cols:  # Remove from numerical if it was added there temporarily
                        self.numerical_cols.remove('cluster')
                    self.pipeline_logger.info("Added 'cluster' feature to dataset and marked as categorical.")
                    # Ensure cluster column is treated as object/category for OHE later
                    X['cluster'] = X['cluster'].astype('category')
                else:
                    self.pipeline_logger.info("Clustering did not add a 'cluster' column (skipped or failed).")
            else:
                self.pipeline_logger.warning("Skipping feature clustering as no numerical columns were found.")

            # 10. Recombine X and y for saving processed data
            self.processed_data = pd.concat([X, y], axis=1)

            # --- Save Processed Data ---
            processed_path = os.path.join(self.visualization_dir, "processed_data.xlsx")
            try:
                self.processed_data.to_excel(processed_path, index=False)
                self.pipeline_logger.info(f"Processed dataset saved to {processed_path}")
            except Exception as write_err:
                self.pipeline_logger.error(f"Failed to save processed data to Excel: {write_err}. Saving as CSV.")
                processed_path_csv = os.path.join(self.visualization_dir, "processed_data.csv")
                try:
                    self.processed_data.to_csv(processed_path_csv, index=False)
                    self.pipeline_logger.info(f"Processed dataset saved to {processed_path_csv}")
                except Exception as csv_err:
                    self.pipeline_logger.error(f"Failed to save processed data to CSV as well: {csv_err}")

            self.pipeline_logger.info(f"--- Data Processing Complete. Final shape: {self.processed_data.shape} ---")
            return self.target_col

        except FileNotFoundError as e:
            self.pipeline_logger.error(f"Data loading failed during processing: {e}", exc_info=True)
            raise  # Re-raise critical errors
        except ValueError as e:
            self.pipeline_logger.error(f"Data processing error: {e}", exc_info=True)
            raise  # Re-raise critical errors
        except Exception as e:
            self.pipeline_logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed unexpectedly during processing. Check 'pipeline.log' for details.")
            raise  # Re-raise unexpected errors

    def select_target_column(self):
        """
        Automatically detects a likely target column based on name, dtype, and cardinality.
        Excludes ID-like columns.
        """
        if self.data is None or self.data.empty:
            raise ValueError("Data is not loaded, cannot select target column.")

        potential_targets = self.data.columns.tolist()
        self.pipeline_logger.debug(f"Initial potential targets: {potential_targets}")

        # --- Exclusion Criteria ---
        # Exclude columns that are likely identifiers (case-insensitive)
        id_like_patterns = ['id', 'key', 'uuid', 'index', 'serial', 'number', 'code', 'pk', 'sk']  # Expanded list
        potential_targets = [
            col for col in potential_targets
            # Exclude if pattern is in name, UNLESS the name IS exactly the pattern (e.g., column named 'number')
            # AND it's not highly unique (might be a meaningful number)
            if not any(pattern in col.lower() for pattern in id_like_patterns) or (
                        col.lower() in id_like_patterns and self.data[col].nunique() / len(self.data) < 0.95)
        ]
        self.pipeline_logger.debug(f"Targets after ID pattern filter: {potential_targets}")

        # Exclude columns with all unique values (likely IDs even if not named like one)
        potential_targets = [
            col for col in potential_targets
            if self.data[col].nunique() < len(self.data)
        ]
        self.pipeline_logger.debug(f"Targets after all-unique filter: {potential_targets}")

        # Exclude columns with only one unique value (no predictive power)
        potential_targets = [
            col for col in potential_targets
            if self.data[col].nunique() > 1
        ]
        self.pipeline_logger.debug(f"Targets after single-unique filter: {potential_targets}")

        # Exclude columns with excessive missing values (e.g., > 50%) - already done in load, but double check
        missing_threshold = 0.5
        potential_targets = [
            col for col in potential_targets
            if self.data[col].isnull().mean() <= missing_threshold
        ]
        self.pipeline_logger.debug(f"Targets after missing value filter: {potential_targets}")

        if not potential_targets:
            # Try relaxing ID filter slightly if nothing is left
            potential_targets = self.data.columns[self.data.isnull().mean() <= missing_threshold].tolist()
            potential_targets = [col for col in potential_targets if self.data[col].nunique() > 1]
            self.pipeline_logger.warning(
                "No ideal targets found after filtering. Considering all columns with <50% missing and >1 unique value.")
            if not potential_targets:
                raise ValueError(
                    "No suitable candidate columns found after filtering for IDs, uniqueness, and missing values.")

        # --- Prioritization Criteria ---
        # 1. Explicit names (case-insensitive)
        common_target_names = ['target', 'label', 'class', 'output', 'result', 'prediction', 'category', 'score', 'y']
        for name in common_target_names:
            for col in potential_targets:
                if name == col.lower():
                    self.pipeline_logger.info(f"Selecting target based on common name match: '{col}'")
                    return col

        # 2. Task Type Heuristics (using _detect_task_type logic)
        classification_candidates = []
        regression_candidates = []

        for col in potential_targets:
            col_type = self._detect_task_type(self.data[col])
            nunique = self.data[col].nunique()
            unique_ratio = nunique / len(self.data) if len(self.data) > 0 else 0

            if col_type == 'classification':
                # Prioritize binary, then lower cardinality
                priority = 0 if nunique == 2 else unique_ratio
                classification_candidates.append((col, priority))
            else:  # Regression
                # Prioritize higher cardinality (more continuous)
                priority = -unique_ratio  # Negative ratio, so sorting ascending works
                regression_candidates.append((col, priority))

        # Sort candidates: lower value is better (binary=0, low ratio=small positive, high ratio regression=large negative)
        classification_candidates.sort(key=lambda x: x[1])
        regression_candidates.sort(key=lambda x: x[1])

        # --- Selection Logic ---
        # Prefer classification targets if available, especially binary ones
        if classification_candidates:
            best_classification = classification_candidates[0][0]
            self.pipeline_logger.info(
                f"Selecting '{best_classification}' as most likely classification target (Priority: {classification_candidates[0][1]:.4f}).")
            return best_classification
        elif regression_candidates:
            best_regression = regression_candidates[0][0]
            self.pipeline_logger.info(
                f"Selecting '{best_regression}' as most likely regression target (Priority: {regression_candidates[0][1]:.4f}).")
            return best_regression

        # Fallback: If no clear winner, pick the last column among candidates? Or raise error.
        # Raising error is safer than arbitrary choice.
        self.pipeline_logger.error(
            f"Could not automatically determine a suitable target column from remaining candidates: {potential_targets}")
        raise ValueError("Could not automatically determine a suitable target column based on heuristics.")

    def prompt_for_target_column(self):
        """ Prompts user to select target column, with auto-detection as fallback suggestion """
        if self.data is None or self.data.empty:
            raise ValueError("Data is not loaded, cannot prompt for target column.")

        print("\n--- Select Target Column ---")
        available_columns = self.data.columns.tolist()
        for i, col in enumerate(available_columns, start=1):
            print(f"{i}. {col}")

        # Try auto-detection first to provide a suggestion
        suggested_target = None
        try:
            suggested_target = self.select_target_column()
            print(f"\nSuggested target based on analysis: '{suggested_target}'")
            prompt = f"Enter the exact target column name (or press Enter to accept suggestion '{suggested_target}'): "
        except ValueError as e:
            self.pipeline_logger.warning(f"Auto-detection failed during prompt: {e}")
            prompt = "Enter the exact target column name: "

        while True:
            user_input = input(prompt).strip()
            if not user_input and suggested_target:
                selected_col = suggested_target
                self.pipeline_logger.info(f"User accepted suggested target: {selected_col}")
                break
            elif user_input in available_columns:
                selected_col = user_input
                self.pipeline_logger.info(f"User manually selected target: {selected_col}")
                break
            elif user_input:
                print(f"Error: Column '{user_input}' not found. Please choose from the list above or check spelling.")
            elif not suggested_target:  # No input and no suggestion
                print("Error: Please enter a valid column name from the list.")

            # If loop continues, reset prompt
            prompt = "Enter the exact target column name: "

        if selected_col not in self.data.columns:
            # This should ideally not happen due to the loop above, but as a safeguard:
            raise ValueError(f"Selected target column '{selected_col}' is not valid.")

        return selected_col

    # =================== Training & Predicting ===================

    def train_predict(self, task_type=None):
        """
        Trains models, computes predictions, metrics, and prepares data for insights.
        Handles feature name extraction and SHAP value computation.
        """
        try:
            self.pipeline_logger.info("--- Training and Prediction Phase Started ---")

            if self.processed_data is None or self.target_col is None:
                raise ValueError("Processed data or target column is not available. Run process_data first.")

            if self.target_col not in self.processed_data.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in processed data.")

            X = self.processed_data.drop(columns=[self.target_col])
            y = self.processed_data[self.target_col]

            if y.empty:
                raise ValueError("Target column 'y' is empty after processing.")
            if X.empty:
                raise ValueError("Feature set 'X' is empty after processing.")

            # --- Determine Task Type ---
            if task_type:
                self.task_type = task_type
                self.pipeline_logger.info(f"Using user-provided task type: {self.task_type}")
            else:
                self.task_type = self._detect_task_type(y)
                self.pipeline_logger.info(f"Auto-detected task type: {self.task_type}")

            # --- Feature Handling (Re-identify from processed X) ---
            self.numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
            # Ensure 'cluster' is treated as categorical if present
            self.categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            if 'cluster' in X.columns:
                if 'cluster' not in self.categorical_cols:
                    self.categorical_cols.append('cluster')
                    self.pipeline_logger.debug("'cluster' added to categorical columns.")
                if 'cluster' in self.numerical_cols:
                    self.numerical_cols.remove('cluster')
                    self.pipeline_logger.debug("'cluster' removed from numerical columns.")
                # Ensure cluster column is treated as object/category for OHE
                X['cluster'] = X['cluster'].astype('category')

            self.pipeline_logger.info(
                f"Features for modeling - Numerical ({len(self.numerical_cols)}): {self.numerical_cols}")
            self.pipeline_logger.info(
                f"Features for modeling - Categorical ({len(self.categorical_cols)}): {self.categorical_cols}")

            # Check for empty feature lists
            if not self.numerical_cols and not self.categorical_cols:
                raise ValueError("No numerical or categorical features identified for modeling.")

            # --- Build Preprocessing Pipeline ---
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Impute again in pipeline for safety
                ('scaler', StandardScaler())
            ])

            # Function to convert all values to strings to ensure uniform type for encoder
            def convert_to_string(X_cat):
                return X_cat.astype(str)

            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),  # Impute again
                ('to_string', FunctionTransformer(convert_to_string)),  # Convert all values to strings
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Dense output
            ])

            # Create the ColumnTransformer - handle cases where one type might be missing
            transformers = []
            if self.numerical_cols:
                transformers.append(('num', numeric_transformer, self.numerical_cols))
            if self.categorical_cols:
                transformers.append(('cat', categorical_transformer, self.categorical_cols))

            if not transformers:
                raise ValueError("No transformers could be created (no numeric or categorical columns?).")

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',  # Keep any columns not explicitly handled (should be none ideally)
                # sparse_threshold=0 # Not needed if OHE is sparse_output=False
            )

            # --- Train/Test Split ---
            # Stratify for classification if target has more than 1 class and enough samples per class
            stratify_opt = None
            if self.task_type == 'classification' and y.nunique() > 1:
                min_class_count = y.value_counts().min()
                # n_splits typically 5 for default CV, so need at least that many per class
                if min_class_count >= 2:  # Basic check, could use n_splits if CV is used later
                    stratify_opt = y
                    self.pipeline_logger.info("Using stratified split.")
                else:
                    self.pipeline_logger.warning(
                        f"Cannot stratify: minimum class count ({min_class_count}) is too low.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_opt
            )
            self.y_test = y_test.copy()  # Store the original test targets
            self.X_test_original = X_test.copy()  # Store original X_test features
            self.pipeline_logger.info(f"Data split: Train shape={X_train.shape}, Test shape={X_test.shape}")

            # --- Define Model Pipelines ---
            # Add class_weight='balanced' for classification if potentially imbalanced
            rf_classifier_params = {'random_state': 42, 'n_jobs': -1}  # Use n_jobs
            if self.task_type == 'classification' and stratify_opt is not None:  # Check if stratification was possible
                rf_classifier_params['class_weight'] = 'balanced'
                self.pipeline_logger.info("Using class_weight='balanced' for RandomForestClassifier.")

            if self.task_type == 'classification':
                self.model_rf = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(**rf_classifier_params))
                ])
                self.model_knn = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))  # Default n_neighbors, use n_jobs
                ])
            elif self.task_type == 'regression':
                self.model_rf = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))  # Use n_jobs
                ])
                self.model_knn = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', KNeighborsRegressor(n_neighbors=5, n_jobs=-1))  # Default n_neighbors, use n_jobs
                ])
            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")

            # --- Fit Models ---
            self.pipeline_logger.info(f"Fitting RandomForest {self.task_type} model...")
            start_fit_rf = time.time()
            self.model_rf.fit(X_train, y_train)
            self.pipeline_logger.info(f"RandomForest fitting complete. Time: {time.time() - start_fit_rf:.2f}s")

            self.pipeline_logger.info("Fitting KNN model...")
            start_fit_knn = time.time()
            self.model_knn.fit(X_train, y_train)
            self.pipeline_logger.info(f"KNN fitting complete. Time: {time.time() - start_fit_knn:.2f}s")

            # --- Get Feature Names AFTER Fitting Preprocessor ---
            # Crucial step: get names from the *fitted* preprocessor in the pipeline
            self.feature_names = []  # Reset feature names
            try:
                fitted_preprocessor = self.model_rf.named_steps['preprocessor']
                feature_names_out = []

                # Get names from transformers
                for name, trans_obj, columns in fitted_preprocessor.transformers_:
                    if trans_obj == 'drop':
                        continue  # Skip dropped columns
                    if name == 'remainder' and fitted_preprocessor.remainder == 'passthrough':
                        # Get original columns not handled by other transformers
                        processed_cols_flat = [col for _, _, cols in fitted_preprocessor.transformers_ if
                                               _ != 'remainder' for col in cols]
                        remainder_cols = [col for col in X_train.columns if col not in processed_cols_flat]
                        feature_names_out.extend(remainder_cols)
                        self.pipeline_logger.debug(f"Remainder columns added: {remainder_cols}")
                        continue

                    # Handle pipelines within ColumnTransformer or direct transformers
                    if hasattr(trans_obj, 'get_feature_names_out'):
                        # If it's a pipeline, get names from the last step
                        if isinstance(trans_obj, Pipeline):
                            last_step_name, last_step_obj = trans_obj.steps[-1]
                            if hasattr(last_step_obj, 'get_feature_names_out'):
                                # Pass original column names to the encoder's get_feature_names_out
                                current_names = last_step_obj.get_feature_names_out(columns)
                                feature_names_out.extend(current_names)
                                self.pipeline_logger.debug(
                                    f"Got {len(current_names)} names from pipeline step '{last_step_name}' in transformer '{name}'")
                            else:  # Scaler, Imputer don't change names relative to input 'columns'
                                feature_names_out.extend(columns)
                                self.pipeline_logger.debug(
                                    f"Got {len(columns)} names (unchanged) from pipeline step '{last_step_name}' in transformer '{name}'")
                        else:  # Direct transformer like OHE
                            current_names = trans_obj.get_feature_names_out(columns)
                            feature_names_out.extend(current_names)
                            self.pipeline_logger.debug(
                                f"Got {len(current_names)} names from direct transformer '{name}'")
                    elif name != 'remainder':  # Transformer doesn't change names (e.g., StandardScaler in 'num')
                        feature_names_out.extend(columns)
                        self.pipeline_logger.debug(
                            f"Got {len(columns)} names (unchanged) from direct transformer '{name}'")

                self.feature_names = feature_names_out
                self.pipeline_logger.info(
                    f"Successfully extracted {len(self.feature_names)} feature names after preprocessing.")
                # self.pipeline_logger.debug(f"Feature names: {self.feature_names}") # Uncomment for debugging

            except Exception as e:
                self.pipeline_logger.error(f"Failed to extract feature names from preprocessor: {e}", exc_info=True)
                # Fallback: Try to get output shape and create placeholders
                try:
                    X_train_transformed_check = self.model_rf.named_steps['preprocessor'].transform(X_train.head(1))
                    num_transformed_features = X_train_transformed_check.shape[1]
                    self.feature_names = [f"feature_{i}" for i in range(num_transformed_features)]
                    self.pipeline_logger.warning(
                        f"Using {num_transformed_features} placeholder feature names due to extraction error.")
                except Exception as transform_err:
                    self.pipeline_logger.error(
                        f"Could not transform data to get feature count for fallback names: {transform_err}. Plots requiring names will likely fail.")
                    self.feature_names = []  # Leave empty if fallback fails

            # --- Make Predictions & Store Processed Test Data ---
            self.pipeline_logger.info("Making predictions on the test set...")
            self.predictions_rf_test = self.model_rf.predict(X_test)
            self.predictions_knn_test = self.model_knn.predict(X_test)

            # Store the processed test set for SHAP/PDP
            try:
                self.X_test_processed = self.model_rf.named_steps['preprocessor'].transform(X_test)
                # Ensure dense format if needed (OHE sparse_output=False should handle this)
                if hasattr(self.X_test_processed, 'toarray'):
                    self.X_test_processed = self.X_test_processed.toarray()
                # Ensure float type for SHAP/PDP
                self.X_test_processed = self.X_test_processed.astype(float)
                self.pipeline_logger.info(
                    f"Stored processed test data (X_test_processed) with shape: {self.X_test_processed.shape}")

                # Verification: Check if processed shape matches feature names length
                if not self.feature_names:
                    self.pipeline_logger.warning("Feature names list is empty. Cannot verify shape consistency.")
                elif self.X_test_processed.shape[1] != len(self.feature_names):
                    self.pipeline_logger.error(
                        f"Shape mismatch! Processed X_test has {self.X_test_processed.shape[1]} columns, but found {len(self.feature_names)} feature names.")
                    # Attempt to fix feature names with placeholders if mismatch detected
                    self.pipeline_logger.warning(
                        "Attempting to fix feature names with placeholders based on processed data shape.")
                    self.feature_names = [f"feature_{i}" for i in range(self.X_test_processed.shape[1])]

            except Exception as proc_err:
                self.pipeline_logger.error(f"Failed to transform X_test for SHAP/PDP: {proc_err}", exc_info=True)
                self.X_test_processed = None  # Set to None if transformation fails

            # --- Log Performance ---
            self.pipeline_logger.info("Evaluating model performance...")
            if self.task_type == 'classification':
                rf_acc = accuracy_score(self.y_test, self.predictions_rf_test)
                knn_acc = accuracy_score(self.y_test, self.predictions_knn_test)
                self.pipeline_logger.info(f"RandomForest Accuracy: {rf_acc:.4f}")
                self.pipeline_logger.info(f"KNN Accuracy: {knn_acc:.4f}")
                try:
                    # Use target names if available from LabelEncoder
                    target_names_report = None
                    if self.target_col in self.label_encoders:
                        target_names_report = self.label_encoders[self.target_col].classes_.astype(str)
                    report = classification_report(self.y_test, self.predictions_rf_test, zero_division=0,
                                                   target_names=target_names_report)
                    self.pipeline_logger.info(f"RF Classification Report:\n{report}")
                except Exception as report_err:
                    self.pipeline_logger.error(f"Could not generate classification report: {report_err}")
            else:  # Regression
                rf_r2 = r2_score(self.y_test, self.predictions_rf_test)
                knn_r2 = r2_score(self.y_test, self.predictions_knn_test)
                rf_mse = mean_squared_error(self.y_test, self.predictions_rf_test)
                knn_mse = mean_squared_error(self.y_test, self.predictions_knn_test)
                self.pipeline_logger.info(f"RandomForest R2: {rf_r2:.4f}, MSE: {rf_mse:.4f}")
                self.pipeline_logger.info(f"KNN R2: {knn_r2:.4f}, MSE: {knn_mse:.4f}")

            # --- Compute SHAP Values ---
            self.shap_values = None  # Reset SHAP values
            if shap_available and self.X_test_processed is not None and self.feature_names:
                self.pipeline_logger.info("Computing SHAP values for RandomForest model...")
                try:
                    core_model = self.model_rf.steps[-1][1]  # Get the actual model

                    # Use appropriate explainer (TreeExplainer for RF)
                    # Provide background data (e.g., transformed X_train sample) for interventional perturbation
                    X_train_proc_sample = self.model_rf.named_steps['preprocessor'].transform(
                        X_train.sample(min(100, len(X_train)), random_state=42))
                    if hasattr(X_train_proc_sample, 'toarray'): X_train_proc_sample = X_train_proc_sample.toarray()
                    X_train_proc_sample = X_train_proc_sample.astype(float)

                    # Limit test samples for SHAP computation to speed up the process
                    max_shap_samples = 500  # Reduced from full test set to 500 samples
                    X_test_shap = None  # Initialize
                    y_test_shap = None  # Initialize
                    if self.X_test_processed.shape[0] > max_shap_samples:
                        self.pipeline_logger.info(
                            f"Limiting SHAP computation to {max_shap_samples} samples for performance")
                        # Take a random sample of indices
                        shap_indices = np.random.choice(self.X_test_processed.shape[0], max_shap_samples, replace=False)
                        X_test_shap = self.X_test_processed[shap_indices]
                        y_test_shap = self.y_test.iloc[shap_indices] if hasattr(self.y_test, 'iloc') else self.y_test[
                            shap_indices]
                    else:
                        X_test_shap = self.X_test_processed
                        y_test_shap = self.y_test

                    self.pipeline_logger.debug(
                        f"Using background data shape {X_train_proc_sample.shape} for SHAP TreeExplainer.")
                    self.pipeline_logger.debug(
                        f"Using test data shape {X_test_shap.shape} for SHAP values calculation.")

                    # Check for data consistency before explaining
                    if X_train_proc_sample.shape[1] != X_test_shap.shape[1]:
                        raise ValueError(
                            f"Background data columns ({X_train_proc_sample.shape[1]}) != Test data columns ({X_test_shap.shape[1]})")

                    explainer = shap.TreeExplainer(
                        core_model,
                        data=X_train_proc_sample,  # Use background data
                        feature_perturbation="interventional",
                        # model_output="raw" # Try 'raw' if prediction probabilities cause issues, might need adjustment
                    )

                    # Add progress indicator for SHAP computation
                    print("\nComputing SHAP values (this may take a while)...")
                    total_samples = X_test_shap.shape[0]
                    batch_size = 50  # Process in smaller batches for progress reporting
                    all_shap_values = []

                    for i in range(0, total_samples, batch_size):
                        # Calculate end index for current batch
                        end_idx = min(i + batch_size, total_samples)
                        # Process current batch
                        batch_shap = explainer(X_test_shap[i:end_idx], check_additivity=False)
                        all_shap_values.append(batch_shap.values)

                        # Print progress
                        progress = min(100, int((end_idx / total_samples) * 100))
                        print(
                            f"\r{progress}%|{'=' * (progress // 2)}{' ' * (50 - (progress // 2))}| {end_idx}/{total_samples}",
                            end="")

                    print("\nSHAP computation complete!")

                    # Combine all batches
                    if not all_shap_values:
                        raise ValueError("SHAP computation returned no values.")

                    if isinstance(all_shap_values[0], np.ndarray):
                        if all_shap_values[0].ndim == 2:  # 2D arrays (regression or binary classification)
                            self.shap_values = np.vstack([sv for sv in all_shap_values])
                        elif all_shap_values[0].ndim == 3:  # 3D arrays (multi-class)
                            self.shap_values = np.vstack([sv for sv in all_shap_values])
                        else:
                            raise ValueError(f"Unexpected SHAP array dimension: {all_shap_values[0].ndim}")
                    elif isinstance(all_shap_values[0], list):  # List of arrays (multi-class)
                        # Initialize with empty lists for each class
                        combined = [[] for _ in range(len(all_shap_values[0]))]
                        # Append values for each class
                        for batch in all_shap_values:
                            for class_idx, class_values in enumerate(batch):
                                combined[class_idx].append(class_values)
                        # Convert lists to arrays
                        self.shap_values = [np.vstack(class_values) for class_values in combined]
                    else:
                        # Fallback for unexpected format
                        raise ValueError(f"Unexpected SHAP value type in list: {type(all_shap_values[0])}")

                    # Handle multi-class output format logging
                    if self.task_type == 'classification':
                        if isinstance(self.shap_values, list) and len(self.shap_values) > 1:
                            self.pipeline_logger.info(
                                f"SHAP values computed as list for {len(self.shap_values)} classes.")
                        elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
                            self.pipeline_logger.info(
                                f"SHAP values computed as 3D array with shape {self.shap_values.shape} (multi-class).")
                        elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 2:
                            self.pipeline_logger.info(
                                f"SHAP values computed as 2D array with shape {self.shap_values.shape} (likely binary classification).")
                        else:
                            self.pipeline_logger.warning(
                                f"Unexpected SHAP values format/shape: {type(self.shap_values)}, shape={getattr(self.shap_values, 'shape', 'N/A')}")

                    self.pipeline_logger.info("SHAP computation completed successfully.")

                except Exception as e:
                    self.pipeline_logger.error(f"Error during SHAP computation: {e}", exc_info=True)
                    self.pipeline_logger.error(traceback.format_exc())  # Log detailed traceback
                    self.shap_values = None  # Ensure it's None on failure
            elif not shap_available:
                self.pipeline_logger.warning("SHAP library not installed. Skipping SHAP computation.")
            else:
                self.pipeline_logger.warning(
                    "Skipping SHAP computation due to missing processed test data or feature names.")

            # --- Generate Insights (Plots etc.) ---
            # This now happens *after* feature names and SHAP values are computed (if possible)
            self.pipeline_logger.info("Generating insights and plots...")
            self._generate_insights(self.y_test, self.predictions_rf_test, self.task_type)

            # --- Save Predictions ---
            predictions_path = os.path.join(self.visualization_dir, "predictions.csv")
            try:
                # Create DataFrame for predictions
                self.predictions = pd.DataFrame({
                    'Actual': self.y_test,
                    'Predicted_RF': self.predictions_rf_test,
                    'Predicted_KNN': self.predictions_knn_test
                }, index=self.y_test.index)  # Preserve index if meaningful

                # If target was encoded, map actual and predictions back to original labels for readability
                if self.target_col in self.label_encoders:
                    le = self.label_encoders[self.target_col]
                    try:
                        # Use inverse_transform, handle potential errors if unseen labels appear in predictions
                        self.predictions['Actual_Original'] = le.inverse_transform(
                            self.predictions['Actual'].astype(int))
                        self.predictions['Predicted_RF_Original'] = le.inverse_transform(
                            self.predictions['Predicted_RF'].astype(int))
                        self.predictions['Predicted_KNN_Original'] = le.inverse_transform(
                            self.predictions['Predicted_KNN'].astype(int))
                    except ValueError as inv_err:
                        self.pipeline_logger.warning(
                            f"Could not inverse transform predictions/actuals: {inv_err}. Saving encoded values.")
                    except Exception as inv_err_other:
                        self.pipeline_logger.warning(
                            f"Unexpected error during inverse transform: {inv_err_other}. Saving encoded values.")

                self.predictions.to_csv(predictions_path, index=True)
                self.pipeline_logger.info(f"Predictions saved to {predictions_path}")
            except Exception as pred_save_err:
                self.pipeline_logger.error(f"Failed to save predictions: {pred_save_err}")

            self.pipeline_logger.info("--- Training and Prediction Phase Finished ---")

        except ValueError as e:
            self.pipeline_logger.error(f"ValueError during training/prediction: {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed during training - {e}")
            raise
        except MemoryError as e:
            self.pipeline_logger.error(
                f"MemoryError during training/prediction: {e}. Try reducing data size or model complexity.",
                exc_info=True)
            print(f"\nERROR: Pipeline failed - Out of memory.")
            raise
        except Exception as e:
            self.pipeline_logger.error(f"Unexpected error during training/prediction: {e}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())  # Log detailed traceback
            print(f"\nERROR: Pipeline failed unexpectedly during training. Check 'pipeline.log'.")
            raise

    # =================== Insights, Visualization & Reporting ===================

    def _generate_insights(self, y_test, y_pred_rf, task_type):
        """ Generates plots and metrics file. Assumes self.feature_names is populated correctly. """
        self.pipeline_logger.info("--- Generating Visual Insights ---")
        # Store metrics calculated here to be saved later
        metrics_to_save = {}
        plot_status = {'confusion': False, 'residuals': False, 'importance': False, 'shap': False, 'pdp': False,
                       'tree': False}

        if not self.feature_names:
            self.pipeline_logger.warning(
                "Feature names are not available. Some plots may fail or have incorrect labels.")

        # --- Basic Performance Plots & Metrics ---
        if task_type == 'classification':
            metrics_to_save['accuracy_rf'] = accuracy_score(y_test, y_pred_rf)
            try:
                # Ensure y_test and y_pred have same length and are 1D
                y_test_flat = np.ravel(y_test)
                y_pred_flat = np.ravel(y_pred_rf)
                if len(y_test_flat) != len(y_pred_flat):
                    self.pipeline_logger.error(
                        "y_test and y_pred_rf have different lengths, cannot generate classification report/matrix.")
                else:
                    target_names = None
                    if self.target_col in self.label_encoders:
                        target_names = self.label_encoders[self.target_col].classes_.astype(str)

                    # Store the report dict in metrics_to_save
                    metrics_to_save['classification_report_rf'] = classification_report(
                        y_test_flat, y_pred_flat, output_dict=True, zero_division=0, target_names=target_names
                    )
                    if self._plot_confusion_matrix(y_test_flat, y_pred_flat):
                        plot_status['confusion'] = True
            except Exception as class_report_err:
                self.pipeline_logger.error(f"Failed to generate classification report/matrix: {class_report_err}",
                                           exc_info=True)

        else:  # Regression
            metrics_to_save['mse_rf'] = mean_squared_error(y_test, y_pred_rf)
            metrics_to_save['r2_rf'] = r2_score(y_test, y_pred_rf)
            if self._plot_residuals(y_test, y_pred_rf):
                plot_status['residuals'] = True

        # --- Advanced Plots (Require feature names and fitted model) ---
        if self.feature_names and self.model_rf:
            if self._plot_feature_importance():
                plot_status['importance'] = True
            # SHAP summary plot generation (uses self.shap_values computed earlier)
            if self._shap_analysis_plot_only():
                plot_status['shap'] = True
            if self._partial_dependence_analysis():
                plot_status['pdp'] = True
            if self._visualize_decision_tree():
                plot_status['tree'] = True
        else:
            self.pipeline_logger.warning(
                "Skipping advanced plots (importance, SHAP, PDP, tree) due to missing feature names or model.")

        self.pipeline_logger.info(f"Plot generation status: {plot_status}")

        # --- Write Metrics to Excel ---
        metrics_filepath = os.path.join(self.visualization_dir, "metrics.xlsx")
        try:
            with pd.ExcelWriter(metrics_filepath) as writer:
                # Add KNN metrics if available
                if self.predictions_knn_test is not None:
                    if task_type == 'classification':
                        metrics_to_save['accuracy_knn'] = accuracy_score(y_test, self.predictions_knn_test)
                    else:
                        metrics_to_save['mse_knn'] = mean_squared_error(y_test, self.predictions_knn_test)
                        metrics_to_save['r2_knn'] = r2_score(y_test, self.predictions_knn_test)

                # Summary Sheet using metrics_to_save
                summary_data = {}
                if task_type == 'classification':
                    summary_data['RF Accuracy'] = [metrics_to_save.get('accuracy_rf')]
                    summary_data['KNN Accuracy'] = [metrics_to_save.get('accuracy_knn')]
                else:
                    summary_data['RF R2'] = [metrics_to_save.get('r2_rf')]
                    summary_data['RF MSE'] = [metrics_to_save.get('mse_rf')]
                    summary_data['KNN R2'] = [metrics_to_save.get('r2_knn')]
                    summary_data['KNN MSE'] = [metrics_to_save.get('mse_knn')]
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary Metrics", index=False)

                # Detailed RF Classification Report Sheet using metrics_to_save
                if task_type == 'classification' and 'classification_report_rf' in metrics_to_save:
                    try:
                        # Convert report dict to DataFrame
                        report_df = pd.DataFrame(metrics_to_save['classification_report_rf']).transpose()
                        report_df.to_excel(writer, sheet_name="RF Class Report")
                    except Exception as report_write_err:
                        self.pipeline_logger.error(
                            f"Failed to write classification report to Excel: {report_write_err}")
                        pd.DataFrame({"Error": [f"Could not write report: {report_write_err}"]}).to_excel(writer,
                                                                                                          sheet_name="RF Class Report",
                                                                                                          index=False)
                elif task_type == 'classification':
                    pd.DataFrame({"Status": ["RF Classification report not generated"]}).to_excel(writer,
                                                                                                  sheet_name="RF Class Report",
                                                                                                  index=False)

            self.pipeline_logger.info(f"Metrics saved to {metrics_filepath}")
        except Exception as excel_err:
            self.pipeline_logger.error(f"Failed to write metrics to Excel file {metrics_filepath}: {excel_err}",
                                       exc_info=True)

        self.pipeline_logger.info("--- Visual Insights Generation Finished ---")

    def _plot_confusion_matrix(self, y_test, y_pred):
        """Plots and saves the confusion matrix."""
        fig = None  # Initialize fig
        try:
            cm = confusion_matrix(y_test, y_pred)
            labels = None
            title = "Confusion Matrix (RF Predictions)"
            if self.task_type == 'classification' and self.target_col in self.label_encoders:
                labels = self.label_encoders[self.target_col].classes_.astype(str)
                title = f"Confusion Matrix (RF Predictions) - Target: {self.target_col}"

            fig = plt.figure(figsize=(max(6, len(labels) // 2 if labels is not None else 1),
                                      max(4, len(labels) // 2 if labels is not None else 1)))  # Dynamic size
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels if labels is not None else 'auto',
                        yticklabels=labels if labels is not None else 'auto')
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()  # Adjust layout
            save_path = os.path.join(self.visualization_dir, "confusion_matrix.png")
            plt.savefig(save_path, bbox_inches='tight')
            self.pipeline_logger.info(f"Confusion matrix saved to {save_path}")
            return True
        except Exception as e:
            self.pipeline_logger.error(f"Failed to plot confusion matrix: {e}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            if fig is not None:
                plt.close(fig)  # Close the specific figure

    def _plot_residuals(self, y_test, y_pred):
        """Plots and saves the residual plot for regression."""
        fig = None
        try:
            residuals = np.array(y_test) - np.array(y_pred)  # Ensure numpy arrays
            fig = plt.figure(figsize=(10, 6))
            # Use predicted values on x-axis, residuals on y-axis
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title("Residual Plot (Actual - Predicted) vs. Predicted (RF)")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.grid(True, linestyle='--', alpha=0.5)
            save_path = os.path.join(self.visualization_dir, "residual_plot.png")
            plt.savefig(save_path, bbox_inches='tight')
            self.pipeline_logger.info(f"Residual plot saved to {save_path}")
            return True
        except Exception as e:
            self.pipeline_logger.error(f"Failed to plot residuals: {e}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            if fig is not None:
                plt.close(fig)

    def _plot_feature_importance(self):
        """ Plots feature importance from the RandomForest model. """
        if not self.feature_names:
            self.pipeline_logger.warning("Cannot plot feature importance: Feature names not available.")
            return False
        if not self.model_rf:
            self.pipeline_logger.warning("Cannot plot feature importance: RF Model not available.")
            return False

        fig = None
        try:
            # Get the core model (classifier/regressor) from the pipeline
            core_model = self.model_rf.steps[-1][1]

            if not hasattr(core_model, "feature_importances_"):
                self.pipeline_logger.warning(
                    f"Model type {type(core_model)} does not support feature_importances_. Skipping plot.")
                return False

            importances = core_model.feature_importances_

            # Ensure importances length matches feature_names length
            if len(importances) != len(self.feature_names):
                self.pipeline_logger.error(
                    f"Feature importance length ({len(importances)}) does not match feature names length ({len(self.feature_names)}). Skipping plot.")
                self.pipeline_logger.debug(f"Importances shape: {importances.shape}")
                self.pipeline_logger.debug(f"Feature Names count: {len(self.feature_names)}")
                return False

            importances_series = pd.Series(importances, index=self.feature_names)
            # Select top N features to plot (e.g., top 20)
            top_n = min(20, len(importances_series))
            if top_n == 0:
                self.pipeline_logger.warning("No feature importances to plot.")
                return False

            importances_sorted = importances_series.nlargest(top_n).sort_values()  # Sort ascending for barh

            fig = plt.figure(figsize=(10, max(6, top_n // 2)))  # Adjust height based on N
            colors = ['red' if 'cluster' in x.lower() else '#1f77b4' for x in
                      importances_sorted.index]  # Highlight cluster
            bars = importances_sorted.plot(kind='barh', color=colors)
            plt.title(f"Top {top_n} Feature Importances (Random Forest)")
            plt.xlabel("Importance Score (Gini impurity reduction or MSE reduction)")
            plt.ylabel("Feature")

            plt.tight_layout()
            save_path = os.path.join(self.visualization_dir, "feature_importance.png")
            plt.savefig(save_path, bbox_inches='tight')  # Use bbox_inches
            self.pipeline_logger.info(f"Feature importance plot saved to {save_path}")
            return True

        except Exception as e:
            self.pipeline_logger.error(f"Failed to plot feature importance: {e}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            if fig is not None:
                plt.close(fig)

    def _shap_analysis_plot_only(self):
        """ Generates and saves SHAP summary plot. Assumes self.shap_values are computed. """
        if self.shap_values is None:
            self.pipeline_logger.warning("SHAP values not available. Skipping SHAP summary plot.")
            return False
        if not self.feature_names:
            self.pipeline_logger.warning("Feature names not available. Skipping SHAP summary plot.")
            return False
        if self.X_test_processed is None:
            self.pipeline_logger.warning(
                "Processed test data (X_test_processed) not available. Skipping SHAP summary plot.")
            return False

        fig = None
        try:
            plt.clf()  # Clear any previous matplotlib state

            # Determine the number of samples used for SHAP calculation
            if isinstance(self.shap_values, np.ndarray):
                shap_sample_count = self.shap_values.shape[0]
            elif isinstance(self.shap_values, list) and self.shap_values:
                shap_sample_count = self.shap_values[0].shape[0]
            else:
                self.pipeline_logger.warning("Could not determine SHAP sample count. Using full X_test_processed.")
                shap_sample_count = self.X_test_processed.shape[0]

            # Ensure X_display matches the number of samples used for SHAP
            if self.X_test_processed.shape[0] < shap_sample_count:
                self.pipeline_logger.warning(
                    f"X_test_processed has fewer rows ({self.X_test_processed.shape[0]}) than SHAP values ({shap_sample_count}). This shouldn't happen. Using available rows.")
                X_display = self.X_test_processed
                # Adjust SHAP values if possible (this is tricky and might indicate a prior error)
                if isinstance(self.shap_values, np.ndarray):
                    self.shap_values = self.shap_values[:X_display.shape[0]]
                elif isinstance(self.shap_values, list):
                    self.shap_values = [sv[:X_display.shape[0]] for sv in self.shap_values]
                shap_sample_count = X_display.shape[0]  # Update count
            elif self.X_test_processed.shape[0] > shap_sample_count:
                self.pipeline_logger.info(
                    f"Using the first {shap_sample_count} rows of X_test_processed to match SHAP values.")
                X_display = self.X_test_processed[:shap_sample_count]
            else:
                X_display = self.X_test_processed

            shap_values_for_plot = self.shap_values  # Use the (potentially adjusted) self.shap_values

            self.pipeline_logger.debug(
                f"SHAP values type: {type(shap_values_for_plot)}, Plot data shape: {X_display.shape}")
            if isinstance(shap_values_for_plot, np.ndarray):
                self.pipeline_logger.debug(f"SHAP values shape: {shap_values_for_plot.shape}")
            elif isinstance(shap_values_for_plot, list):
                self.pipeline_logger.debug(f"SHAP values list length: {len(shap_values_for_plot)}")
                if len(shap_values_for_plot) > 0:
                    self.pipeline_logger.debug(f"SHAP values first element shape: {shap_values_for_plot[0].shape}")

            # Handle multi-class SHAP values (list or 3D array)
            target_class_index = 0  # Default to class 0 or regression
            target_class_name = " (Regression or Class 0)"
            if self.task_type == 'classification':
                n_classes = 1
                if isinstance(shap_values_for_plot, list):
                    n_classes = len(shap_values_for_plot)
                elif isinstance(shap_values_for_plot, np.ndarray) and shap_values_for_plot.ndim == 3:
                    n_classes = shap_values_for_plot.shape[2]

                if n_classes > 1:
                    # For binary classification, always use class 0 to avoid reshape errors
                    target_class_index = 0
                    if self.target_col in self.label_encoders:
                        try:
                            target_class_name = f" (Class: {self.label_encoders[self.target_col].classes_[target_class_index]})"
                        except IndexError:
                            target_class_name = f" (Class Index {target_class_index})"
                    else:
                        target_class_name = f" (Class Index {target_class_index})"

                    self.pipeline_logger.info(
                        f"Multi-class SHAP values detected. Plotting summary for class index {target_class_index}{target_class_name}.")

                    if isinstance(shap_values_for_plot, list):
                        if target_class_index < len(shap_values_for_plot):
                            shap_values_for_plot = shap_values_for_plot[target_class_index]
                        else:
                            self.pipeline_logger.warning(
                                f"Target class index {target_class_index} out of bounds for SHAP list. Using class 0.")
                            target_class_index = 0
                            shap_values_for_plot = shap_values_for_plot[0]
                    elif shap_values_for_plot.ndim == 3:
                        if target_class_index < shap_values_for_plot.shape[2]:
                            shap_values_for_plot = shap_values_for_plot[:, :, target_class_index]
                        else:
                            self.pipeline_logger.warning(
                                f"Target class index {target_class_index} out of bounds for SHAP array shape {shap_values_for_plot.shape}. Using class 0.")
                            target_class_index = 0
                            shap_values_for_plot = shap_values_for_plot[:, :, 0]
                elif isinstance(shap_values_for_plot, np.ndarray) and shap_values_for_plot.ndim == 2:
                    # Binary classification might sometimes return only one set of values (for class 1)
                    self.pipeline_logger.info(
                        "Assuming SHAP values are for the positive class in binary classification.")
                    target_class_name = " (Positive Class)"
                else:
                    # Single class or unexpected format
                    target_class_name = " (Class 0 or Unknown)"

            # Ensure feature names match the number of SHAP features
            if shap_values_for_plot.shape[1] != len(self.feature_names):
                self.pipeline_logger.error(
                    f"SHAP values feature count ({shap_values_for_plot.shape[1]}) mismatches feature names count ({len(self.feature_names)}). Skipping SHAP plot.")
                return False

            # Create DataFrame for better feature name handling in plot
            X_display_df = pd.DataFrame(X_display, columns=self.feature_names)

            # Ensure X_display_df columns match shap_values_for_plot columns
            if isinstance(shap_values_for_plot, np.ndarray) and X_display_df.shape[1] != shap_values_for_plot.shape[1]:
                self.pipeline_logger.warning(
                    f"Column count mismatch: X_display_df has {X_display_df.shape[1]} columns, shap_values_for_plot has {shap_values_for_plot.shape[1]} columns. Adjusting feature names...")
                # Adjust feature names to match SHAP values columns
                if X_display_df.shape[1] > shap_values_for_plot.shape[1]:
                    # If X_display_df has more columns, trim it
                    X_display_df = X_display_df.iloc[:, :shap_values_for_plot.shape[1]]
                else:
                    # If SHAP values have more columns, trim them
                    shap_values_for_plot = shap_values_for_plot[:, :X_display_df.shape[1]]

            # Limit to top 20 features for faster plotting
            max_display_features = 20
            plot_feature_names = self.feature_names  # Use full list initially
            if shap_values_for_plot.shape[1] > max_display_features:
                self.pipeline_logger.info(f"Limiting SHAP plot to top {max_display_features} features for performance")

                # Calculate mean absolute SHAP value for each feature
                feature_importance = np.mean(np.abs(shap_values_for_plot), axis=0)
                top_indices = np.argsort(feature_importance)[-max_display_features:]

                # Subset both SHAP values and feature names
                shap_values_for_plot = shap_values_for_plot[:, top_indices]
                X_display_df = X_display_df.iloc[:, top_indices]

                # Update feature names list for the plot
                plot_feature_names = [self.feature_names[i] for i in top_indices]
                X_display_df.columns = plot_feature_names  # Update DF columns for plot

            # Generate the plot with optimized settings
            fig = plt.figure(figsize=(10, 8))  # Create a figure context with fixed size
            shap.summary_plot(
                shap_values_for_plot,
                X_display_df,  # Pass DataFrame with correct column names
                feature_names=plot_feature_names,  # Explicitly pass potentially subsetted names
                show=False,
                max_display=max_display_features,  # Limit displayed features
                plot_size=None  # Let summary_plot manage size within the figure context
            )
            plt.title(f"SHAP Summary Plot{target_class_name}")
            save_path = os.path.join(self.visualization_dir, "shap_summary.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=100)  # Lower DPI for faster saving
            self.pipeline_logger.info(f"SHAP summary plot saved to {save_path}")
            return True

        except Exception as e:
            self.pipeline_logger.error(f"SHAP summary plot generation failed: {str(e)}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            # Ensure plot context is closed properly
            plt.close('all')  # Close all figures just in case

    def _partial_dependence_analysis(self):
        """ Generates PDP plots for top features. Assumes self.feature_names, model, X_test_processed are populated. """
        self.pipeline_logger.info("Generating Partial Dependence Plots...")
        if PartialDependenceDisplay is None:
            self.pipeline_logger.warning(
                "PartialDependenceDisplay not available (check scikit-learn version). Skipping PDP.")
            return False
        if not self.feature_names:
            self.pipeline_logger.warning("Cannot plot Partial Dependence: Feature names not available.")
            return False
        if not self.model_rf:
            self.pipeline_logger.warning("Cannot plot Partial Dependence: RandomForest model not available.")
            return False
        if self.X_test_processed is None:
            self.pipeline_logger.warning("Cannot plot Partial Dependence: Processed test data not available.")
            return False

        fig = None  # Initialize fig
        try:
            core_model = self.model_rf.steps[-1][1]

            if not hasattr(core_model, "feature_importances_"):
                self.pipeline_logger.warning(
                    f"Model type {type(core_model)} does not support feature_importances_. Cannot determine top features for PDP. Skipping.")
                return False

            importances = core_model.feature_importances_
            if len(importances) != len(self.feature_names):
                self.pipeline_logger.error(
                    f"Feature importance length ({len(importances)}) != feature names length ({len(self.feature_names)}) in PDP. Skipping.")
                return False

            # Select top N features (e.g., top 6 for a 2x3 grid) for PDP
            top_n_pdp = min(6, len(self.feature_names))
            if top_n_pdp == 0:
                self.pipeline_logger.warning("No features available for PDP.")
                return False

            # Get indices of top features based on importance scores
            # These indices correspond to the columns in the *transformed* data
            all_feature_indices = np.argsort(importances)[::-1]  # All indices, descending order by importance

            # Filter out potentially problematic features AND prioritize numeric
            filtered_feature_indices = []
            numeric_indices = []
            other_indices = []

            for idx in all_feature_indices:
                feature_name = self.feature_names[idx]
                # Skip features that are likely to cause problems
                if "__MISSING__" in feature_name:
                    self.pipeline_logger.info(
                        f"Skipping potential problematic feature for PDP: {feature_name} (contains __MISSING__)")
                    continue
                # Heuristic for season-related features (adjust as needed)
                if "season" in feature_name.lower() and any(
                        s in feature_name for s in ["1 Season", "2 Seasons", "3 Seasons"]):
                    self.pipeline_logger.info(
                        f"Skipping potential problematic feature for PDP: {feature_name} (Low-count Season)")
                    continue

                # Check if the original feature was numeric (before OHE)
                # This requires mapping back, which is complex. Simpler: check if name is in self.numerical_cols
                # OR if it doesn't contain '___' which is typical for OHE.
                is_likely_numeric_or_simple_cat = False
                if '___' not in feature_name:  # Likely original numeric or simple categorical
                    # Check if it was originally numeric
                    original_col_name = feature_name.split('___')[0]  # Get base name
                    if original_col_name in self.numerical_cols:
                        is_likely_numeric_or_simple_cat = True
                    # Also allow simple categorical features if needed
                    elif original_col_name in self.categorical_cols and self.processed_data[
                        original_col_name].nunique() < 20:  # Allow low-cardinality cats
                        is_likely_numeric_or_simple_cat = True

                if is_likely_numeric_or_simple_cat:
                    numeric_indices.append(idx)
                else:
                    other_indices.append(idx)  # Likely OHE features

            # Prioritize numeric/simple features, then add others if needed
            filtered_feature_indices.extend(numeric_indices)
            remaining_needed = top_n_pdp - len(filtered_feature_indices)
            if remaining_needed > 0:
                filtered_feature_indices.extend(other_indices[:remaining_needed])

            # Take only the top N filtered features
            top_features_transformed_indices = filtered_feature_indices[:top_n_pdp]
            top_features_names = [self.feature_names[i] for i in top_features_transformed_indices]

            if not top_features_transformed_indices:
                self.pipeline_logger.warning("No suitable features found for PDP after filtering.")
                return False

            self.pipeline_logger.info(
                f"Generating PDP for filtered top {len(top_features_transformed_indices)} features: {top_features_names}")
            self.pipeline_logger.debug(f"Using feature indices in transformed data: {top_features_transformed_indices}")

            # Use the processed test data (or a sample if very large) for PDP calculation
            # Reduce sample size for faster processing
            sample_size_pdp = min(500, self.X_test_processed.shape[0])  # Reduced from 1000 to 500
            pdp_indices = np.random.choice(self.X_test_processed.shape[0], sample_size_pdp, replace=False)
            X_pdp_data = self.X_test_processed[pdp_indices, :]
            self.pipeline_logger.info(f"Using a sample of {sample_size_pdp} points for PDP calculation.")

            # Determine target class for multi-class classification
            pdp_target_class = None
            title_suffix = ""
            if self.task_type == 'classification':
                unique_classes = np.unique(self.y_test)
                num_classes = len(unique_classes)
                self.pipeline_logger.info(f"Found {num_classes} unique classes in y_test: {unique_classes}")

                # Check if the model is a binary classifier
                is_binary_classifier = False
                try:
                    # Check if the model has predict_proba method
                    if hasattr(core_model, 'predict_proba'):
                        # Try to get a prediction to check its shape
                        sample_pred = core_model.predict_proba(X_pdp_data[:1])
                        is_binary_classifier = sample_pred.shape[1] == 2
                        self.pipeline_logger.info(
                            f"Model prediction shape: {sample_pred.shape}, binary classifier: {is_binary_classifier}")
                    else:
                        self.pipeline_logger.info(
                            "Model does not have predict_proba method, cannot determine if binary classifier")
                except Exception as e:
                    self.pipeline_logger.warning(f"Could not determine if model is binary classifier: {e}")

                if num_classes > 1:
                    # For multi-class classification, use class index 0
                    pdp_target_class = 0
                    title_suffix = f" (Target Class {pdp_target_class})"
                    self.pipeline_logger.info(f"PDP: Plotting for target class {pdp_target_class}.")
                else:
                    # For single class in test data, don't specify target class to avoid reshape errors
                    self.pipeline_logger.info(
                        "PDP: Only one class detected, plotting standard PDP without target class.")

            # Generate PDP plots
            pdp_kind = 'average'  # Use 'average' for all cases to avoid reshape errors
            n_cols_grid = 3  # Arrange plots in grid
            n_rows_grid = (len(top_features_transformed_indices) + n_cols_grid - 1) // n_cols_grid

            # Create figure and axes for the grid
            fig, ax = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid),
                                   constrained_layout=True)
            ax = ax.flatten() if hasattr(ax, 'flatten') else [ax]  # Handle single subplot case

            # Try to generate PDP plots for each feature individually to avoid total failure
            successful_features = 0
            for i, feature_idx in enumerate(top_features_transformed_indices):
                if i >= len(ax):
                    break  # Stop if we run out of axes

                feature_name = self.feature_names[feature_idx]
                error_message = None  # Store potential error message

                # --- Check for single unique value in the sample ---
                try:
                    feature_values_in_sample = X_pdp_data[:, feature_idx]
                    unique_values_in_sample = np.unique(feature_values_in_sample)
                    if len(unique_values_in_sample) <= 1:
                        self.pipeline_logger.warning(
                            f"Skipping PDP for '{feature_name}': Only {len(unique_values_in_sample)} unique value(s) in the PDP sample data.")
                        error_message = f"Skipped: {feature_name}\n(Only 1 unique value in sample)"
                        # Continue to the error display logic below
                except Exception as check_err:
                    self.pipeline_logger.warning(
                        f"Error checking unique values for PDP feature '{feature_name}': {check_err}")
                    error_message = f"PDP Error for\n{feature_name}\n(Check unique values)"
                    # Continue to the error display logic below
                # --- End Check ---

                # If no error message yet, try generating the plot
                if error_message is None:
                    try:
                        # Determine arguments based on target class
                        pdp_args = {
                            'estimator': core_model,
                            'X': X_pdp_data,
                            'features': [feature_idx],
                            'feature_names': self.feature_names,
                            'kind': pdp_kind,
                            'random_state': 42,
                            'ax': ax[i]
                        }
                        if pdp_target_class is not None:
                            pdp_args['target'] = pdp_target_class

                        # Generate PDP
                        feature_display = PartialDependenceDisplay.from_estimator(**pdp_args)
                        ax[i].set_title(f"PDP: {feature_name}", fontsize=10)  # Set clearer title
                        successful_features += 1
                        continue  # Go to the next feature if successful

                    except ValueError as ve:
                        # Check for the specific reshape error or other common ValueErrors
                        if "cannot reshape array" in str(ve) or "contains non-numeric data" in str(
                                ve) or "could not convert" in str(ve) or "Feature not suitable for PDP" in str(ve):
                            self.pipeline_logger.warning(
                                f"ValueError (likely type/reshape) for PDP of '{feature_name}': {ve}")
                            error_message = f"PDP Failed:\n{feature_name}\n(Type/Reshape Error)"
                        else:
                            self.pipeline_logger.warning(f"Generic ValueError for PDP of '{feature_name}': {ve}")
                            error_message = f"PDP Failed:\n{feature_name}\n(ValueError)"
                    except IndexError as ie:
                        self.pipeline_logger.warning(f"IndexError during PDP for '{feature_name}': {ie}")
                        error_message = f"PDP Failed:\n{feature_name}\n(Index Error)"
                    except Exception as feature_err:
                        self.pipeline_logger.warning(
                            f"Unexpected error generating PDP for '{feature_name}': {feature_err}", exc_info=True)
                        error_message = f"PDP Failed:\n{feature_name}\n(Error)"

                # --- Display Error on Axis if error_message is set ---
                if error_message:
                    ax[i].clear()
                    ax[i].text(0.5, 0.5, error_message,
                               horizontalalignment='center', verticalalignment='center',
                               transform=ax[i].transAxes, fontsize=9, color='red')
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])

            # Remove empty subplots if any
            for i in range(len(top_features_transformed_indices), len(ax)):
                fig.delaxes(ax[i])

            if successful_features == 0:
                self.pipeline_logger.error("All PDP plots failed to generate.")
                plt.close(fig)
                return False

            fig.suptitle(f"Partial Dependence Plots{title_suffix}", fontsize=16)

            pdp_path = os.path.join(self.visualization_dir, "partial_dependence_plots.png")
            fig.savefig(pdp_path, bbox_inches='tight')
            self.pipeline_logger.info(f"Partial dependence plots saved to {pdp_path}")
            return True

        except ValueError as ve:
            # Specific error handling, e.g., for categorical features if PDP fails
            self.pipeline_logger.error(
                f"ValueError during Partial Dependence calculation: {ve}. Check feature types/indices.", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())

            # Create a simple fallback plot with error message
            try:
                fallback_fig, fallback_ax = plt.subplots(figsize=(10, 6))
                fallback_ax.text(0.5, 0.5, f"Partial Dependence Plot generation failed:\n{str(ve)}",
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=fallback_ax.transAxes, fontsize=12)
                fallback_ax.set_xticks([])
                fallback_ax.set_yticks([])
                fallback_ax.set_title("PDP Error", fontsize=14)

                pdp_path = os.path.join(self.visualization_dir, "partial_dependence_plots.png")
                fallback_fig.savefig(pdp_path, bbox_inches='tight')
                plt.close(fallback_fig)
                self.pipeline_logger.info(f"Fallback error message plot saved to {pdp_path}")
            except Exception as fallback_err:
                self.pipeline_logger.error(f"Even fallback plot failed: {fallback_err}")

            return False
        except Exception as e:
            self.pipeline_logger.error(f"Partial dependence plot generation failed: {str(e)}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            if fig is not None:
                plt.close(fig)  # Close the figure explicitly

    def _visualize_decision_tree(self):
        """ Visualizes the first tree of the RandomForest model. """
        self.pipeline_logger.info("Visualizing example decision tree...")
        if not self.feature_names:
            self.pipeline_logger.warning("Cannot visualize Decision Tree: Feature names not available.")
            return False
        if not self.model_rf:
            self.pipeline_logger.warning("Cannot visualize Decision Tree: RandomForest model not available.")
            return False

        fig = None
        try:
            core_model = self.model_rf.steps[-1][1]

            if not isinstance(core_model, (RandomForestClassifier, RandomForestRegressor)):
                self.pipeline_logger.info(
                    f"Decision tree visualization only applicable to RandomForest models, not {type(core_model)}. Skipping.")
                return False

            if not hasattr(core_model, "estimators_") or not core_model.estimators_:
                self.pipeline_logger.warning("RandomForest model has no estimators (trees). Skipping visualization.")
                return False

            tree_to_plot = core_model.estimators_[0]
            fig = plt.figure(figsize=(25, 15))  # Adjust size as needed

            # Determine class names if classification
            class_names = None
            if self.task_type == 'classification':
                if self.target_col in self.label_encoders:
                    class_names = self.label_encoders[self.target_col].classes_.astype(str)
                elif hasattr(core_model, 'classes_'):
                    class_names = core_model.classes_.astype(str)

            plot_tree(
                tree_to_plot,
                feature_names=self.feature_names,
                class_names=class_names,
                filled=True,
                impurity=True,
                rounded=True,
                max_depth=3,  # Limit depth for readability
                fontsize=10
            )
            plt.title("Example Decision Tree (First Tree, Max Depth 3)", fontsize=16)
            save_path = os.path.join(self.visualization_dir, "decision_tree.png")
            plt.savefig(save_path, bbox_inches='tight')
            self.pipeline_logger.info(f"Decision tree visualization saved to {save_path}")
            return True

        except Exception as e:
            self.pipeline_logger.error(f"Decision tree visualization failed: {str(e)}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            return False
        finally:
            if fig is not None:
                plt.close(fig)

    # =================== Dashboard ===================

    def load_image(self, path):
        """ Loads an image file and encodes it for Dash display, handling errors. """
        img_src = None
        try:
            if not os.path.exists(path):
                self.pipeline_logger.warning(f"Image file not found for dashboard: {path}")
                return None  # Return None if file doesn't exist

            with open(path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('ascii')
            img_src = f'data:image/png;base64,{encoded_image}'
            self.pipeline_logger.debug(f"Successfully loaded image: {path}")
            return img_src
        except Exception as e:
            self.pipeline_logger.error(f"Failed to load or encode image {path}: {e}")
            return None  # Return None on error

    def create_dashboard(self):
        """ Builds and runs the Dash application in a separate thread. """
        if self.dashboard_running:
            self.pipeline_logger.warning("Dashboard is already running or starting. Skipping new instance.")
            return

        try:
            self.pipeline_logger.info("--- Creating Dashboard ---")
            app = Dash(__name__, suppress_callback_exceptions=True)
            server = app.server  # For deployment

            # --- Check for Essential Data ---
            if self.processed_data is None:
                error_message = "Processed data is not available. Cannot create dashboard."
                self.pipeline_logger.error(f"Dashboard creation failed: {error_message}")
                app.layout = html.Div([html.H1("Error"), html.P(error_message)])
                # Optionally run the error app here if desired
                # ...
                return  # Stop dashboard creation if essential data is missing

            # --- Load Images (with checks) ---
            self.pipeline_logger.info("Loading images for dashboard...")
            feature_importance_img = self.load_image(os.path.join(self.visualization_dir, "feature_importance.png"))
            shap_summary_img = self.load_image(os.path.join(self.visualization_dir, "shap_summary.png"))
            pdp_img = self.load_image(os.path.join(self.visualization_dir, "partial_dependence_plots.png"))
            confusion_matrix_img = self.load_image(os.path.join(self.visualization_dir, "confusion_matrix.png"))
            residual_plot_img = self.load_image(os.path.join(self.visualization_dir, "residual_plot.png"))
            decision_tree_img = self.load_image(os.path.join(self.visualization_dir, "decision_tree.png"))

            # --- Build Tabs ---
            # Pass necessary data/images to helper methods
            data_overview_layout = self._build_data_overview_tab()
            feature_analysis_layout = self._build_feature_analysis_tab(feature_importance_img, shap_summary_img,
                                                                       pdp_img)
            model_performance_layout = self._build_model_performance_tab(confusion_matrix_img, residual_plot_img)
            clustering_layout = self._build_clustering_tab()
            predictions_layout = self._build_predictions_tab()
            shap_layout = self._build_shap_tab()  # SHAP Bar Chart
            tree_layout = self._build_tree_tab(decision_tree_img)

            # --- Assemble App Layout ---
            app.layout = html.Div([
                html.H1("Automated Data Science Pipeline Dashboard",
                        style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Tabs(id="dashboard-tabs", children=[
                    dcc.Tab(label='Data Overview', value='tab-data', children=[data_overview_layout]),
                    dcc.Tab(label='Feature Analysis', value='tab-features', children=[feature_analysis_layout]),
                    dcc.Tab(label='Model Performance', value='tab-performance', children=[model_performance_layout]),
                    dcc.Tab(label='Clustering', value='tab-clustering', children=[clustering_layout]),
                    dcc.Tab(label='Predictions', value='tab-predictions', children=[predictions_layout]),
                    dcc.Tab(label='SHAP Importance', value='tab-shap', children=[shap_layout]),
                    dcc.Tab(label='Decision Tree', value='tab-tree', children=[tree_layout]),
                ])
            ])

            # --- Define Callbacks ---
            self._define_dashboard_callbacks(app)

            # --- Run the dashboard in a separate thread ---
            def run_server():
                self.pipeline_logger.info("Starting Dash server on http://127.0.0.1:8050 ...")
                self.dashboard_running = True
                try:
                    # Use host='0.0.0.0' to make it accessible on the network
                    app.run(debug=False, port=8050, use_reloader=False, host='0.0.0.0')
                except Exception as server_err:
                    self.pipeline_logger.error(f"Dash server failed to start or crashed: {server_err}", exc_info=True)
                finally:
                    self.dashboard_running = False  # Mark as not running if server stops
                    self.pipeline_logger.info("Dash server stopped.")

            dashboard_thread = threading.Thread(target=run_server, daemon=True)
            dashboard_thread.start()
            self.pipeline_logger.info(
                "Dashboard thread started. Access at http://<your-ip-address>:8050 or http://127.0.0.1:8050")
            print("\nDashboard starting... Access at http://127.0.0.1:8050 (or your local IP)")
            print("(Allow a few seconds for the server to initialize)")

        except Exception as e:
            self.pipeline_logger.error(f"Dashboard creation failed: {e}", exc_info=True)
            self.pipeline_logger.error(traceback.format_exc())
            print(f"\nERROR: Failed to create dashboard. Check 'pipeline.log'.")
            # Don't raise here if the main pipeline should continue

    # --- Dashboard Tab Builder Methods ---

    def _build_data_overview_tab(self):
        self.pipeline_logger.debug("Building Data Overview tab...")
        if self.processed_data is None: return html.Div("Processed data not available.")

        data_for_overview = self.processed_data.copy()
        # Exclude high cardinality columns from dropdown for performance
        plot_candidates = [c for c in data_for_overview.columns if data_for_overview[c].nunique() < 1000]
        if not plot_candidates and len(data_for_overview.columns) > 0:
            plot_candidates = data_for_overview.columns.tolist()  # Fallback
        default_plot_col = plot_candidates[0] if plot_candidates else None

        feature_selector = dcc.Dropdown(
            id='column-selector',
            options=[{'label': col, 'value': col} for col in plot_candidates],
            value=default_plot_col,
            clearable=False,
            style={'marginBottom': '10px'}
        ) if default_plot_col else dcc.Dropdown(id='column-selector', disabled=True,
                                                placeholder="No suitable columns for distribution plot")

        # Correlation Heatmap
        heatmap_graph = html.P("Not enough numeric columns (>1) to generate a correlation heatmap.")
        try:
            numeric_overview_cols = data_for_overview.select_dtypes(include=np.number).columns
            if len(numeric_overview_cols) > 1:
                corr_matrix = data_for_overview[numeric_overview_cols].corr().round(2)
                heatmap_fig = px.imshow(
                    corr_matrix, text_auto=True,  # Show values on heatmap
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title="Correlation Heatmap (Numeric Features)"
                )
                heatmap_fig.update_layout(height=max(600, 40 * len(numeric_overview_cols)))  # Adjust height
                heatmap_graph = dcc.Graph(id='correlation-heatmap', figure=heatmap_fig)
        except Exception as hm_err:
            self.pipeline_logger.error(f"Failed to generate correlation heatmap: {hm_err}")
            heatmap_graph = html.P(f"Error generating correlation heatmap: {hm_err}")

        return html.Div([
            html.H3("Data Overview"),
            html.Div("Select a feature to visualize its distribution:", style={'marginTop': '10px'}),
            feature_selector,
            dcc.Graph(id='histogram'),
            html.Br(),
            html.H4("Descriptive Statistics"),
            html.Div(id='stats-table'),
            html.H3("Correlation Matrix", style={'marginTop': '20px'}),
            heatmap_graph,
            html.H3("Processed Data Sample", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='processed-data-table',
                data=data_for_overview.head(20).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in data_for_overview.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                            'whiteSpace': 'normal'},
                tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for
                              row in data_for_overview.head(20).to_dict('records')],
                tooltip_duration=None
            )
        ], style={'padding': '20px'})

    def _build_feature_analysis_tab(self, feature_importance_img, shap_summary_img, pdp_img):
        self.pipeline_logger.debug("Building Feature Analysis tab...")
        img_style = {'maxWidth': '90%', 'height': 'auto', 'margin': 'auto', 'display': 'block', 'marginBottom': '20px'}

        return html.Div([
            html.H3("Feature Analysis"),
            html.Hr(),
            html.H4("Feature Importance (RandomForest)"),
            html.Img(id='feature-importance-plot', src=feature_importance_img,
                     style=img_style) if feature_importance_img else html.P(
                "Feature importance plot not generated or found."),
            html.Hr(),
            html.H4("SHAP Summary Plot"),
            html.Img(id='shap-summary-plot', src=shap_summary_img, style=img_style) if shap_summary_img else html.P(
                "SHAP summary plot not generated or found."),
            html.Hr(),
            html.H4("Partial Dependence Plots (Top Features)"),
            html.Img(id='pdp-plot', src=pdp_img, style=img_style) if pdp_img else html.P(
                "Partial Dependence Plots not generated or found."),
        ], style={'padding': '20px'})

    def _build_model_performance_tab(self, confusion_matrix_img, residual_plot_img):
        self.pipeline_logger.debug("Building Model Performance tab...")
        perf_children = [html.H3("Model Performance")]

        # --- Check if necessary data is available ---
        if self.y_test is None or self.predictions_rf_test is None or self.predictions_knn_test is None:
            perf_children.append(html.P(
                "Model performance data not available (models might not have been trained or predictions are missing)."))
            return html.Div(perf_children, style={'padding': '20px'})

        try:
            # --- Calculate metrics directly here ---
            rf_metric = None
            knn_metric = None
            metric_label = "Metric"
            plot_title = "Performance Plot"
            perf_img = None
            bullet_charts = []  # Initialize list for bullet charts

            if self.task_type == 'classification':
                rf_metric = accuracy_score(self.y_test, self.predictions_rf_test)
                knn_metric = accuracy_score(self.y_test, self.predictions_knn_test)
                metric_label = "Accuracy"
                plot_title = "Confusion Matrix (Random Forest)"
                perf_img = confusion_matrix_img
                self.pipeline_logger.info("Task is classification, skipping bullet charts.")

            elif self.task_type == 'regression':
                rf_metric = r2_score(self.y_test, self.predictions_rf_test)
                knn_metric = r2_score(self.y_test, self.predictions_knn_test)
                metric_label = "R2 Score"
                plot_title = "Residual Plot (Random Forest)"
                perf_img = residual_plot_img

                # --- START: Bullet Chart Generation for Regression ---
                self.pipeline_logger.info("Task is regression, generating bullet charts for average performance.")
                try:
                    # Use nanmean for robustness against potential NaNs
                    mean_actual = np.nanmean(self.y_test)
                    mean_pred_rf = np.nanmean(self.predictions_rf_test)
                    mean_pred_knn = np.nanmean(self.predictions_knn_test)

                    # Determine a reasonable range for the gauge axis
                    all_vals = np.concatenate([
                        self.y_test.values.flatten(),  # Use .values for Series
                        self.predictions_rf_test.flatten(),
                        self.predictions_knn_test.flatten()
                    ])
                    min_val = np.nanmin(all_vals)
                    max_val = np.nanmax(all_vals)
                    padding = (max_val - min_val) * 0.1  # Add 10% padding
                    gauge_range = [min_val - padding, max_val + padding]

                    # Create RF Bullet Chart
                    fig_bullet_rf = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=mean_pred_rf,
                        delta={'reference': mean_actual, 'relative': False, 'valueformat': '.3f'},
                        title={'text': "Avg. Predicted (RF) vs. Avg. Actual", 'font': {'size': 16}},
                        number={'valueformat': '.3f'},
                        gauge={
                            'axis': {'range': gauge_range, 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#1f77b4"},  # Plotly blue
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                # Optional: Add steps/ranges if desired, e.g., +/- std dev
                                # {'range': [mean_actual - std_actual, mean_actual + std_actual], 'color': 'lightgray'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': mean_actual  # Target line at average actual
                            }
                        }
                    ))
                    fig_bullet_rf.update_layout(height=250,
                                                margin={'t': 50, 'b': 50, 'l': 50, 'r': 50})  # Adjust layout

                    # Create KNN Bullet Chart
                    fig_bullet_knn = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=mean_pred_knn,
                        delta={'reference': mean_actual, 'relative': False, 'valueformat': '.3f'},
                        title={'text': "Avg. Predicted (KNN) vs. Avg. Actual", 'font': {'size': 16}},
                        number={'valueformat': '.3f'},
                        gauge={
                            'axis': {'range': gauge_range, 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#ff7f0e"},  # Plotly orange
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [],  # Optional steps
                            'threshold': {
                                'line': {'color': "red", 'width': 3},
                                'thickness': 0.75,
                                'value': mean_actual  # Target line at average actual
                            }
                        }
                    ))
                    fig_bullet_knn.update_layout(height=250,
                                                 margin={'t': 50, 'b': 50, 'l': 50, 'r': 50})  # Adjust layout

                    # Add charts to the layout
                    bullet_charts = [
                        html.Div([
                            html.H4("Average Performance Overview (Regression)", style={'textAlign': 'center'}),
                            html.Div(dcc.Graph(id='bullet-rf', figure=fig_bullet_rf),
                                     style={'width': '48%', 'display': 'inline-block'}),
                            html.Div(dcc.Graph(id='bullet-knn', figure=fig_bullet_knn),
                                     style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
                        ], style={'marginTop': '20px', 'marginBottom': '20px'})
                    ]
                    self.pipeline_logger.info("Successfully generated bullet charts.")

                except Exception as bullet_err:
                    self.pipeline_logger.error(f"Failed to generate bullet charts: {bullet_err}", exc_info=True)
                    bullet_charts = [html.P(f"Error generating bullet charts: {bullet_err}")]
                # --- END: Bullet Chart Generation ---

            # Scatter plots for predictions vs actual
            scatter_plots = html.P("Could not generate prediction scatter plots.")
            try:
                # Ensure lengths match for plotting (robustness)
                min_len = min(len(self.y_test), len(self.predictions_rf_test), len(self.predictions_knn_test))
                # Ensure indices align if y_test is a Series
                y_test_plot = self.y_test[:min_len].values if isinstance(self.y_test, pd.Series) else self.y_test[
                                                                                                      :min_len]
                pred_rf_plot = self.predictions_rf_test[:min_len]
                pred_knn_plot = self.predictions_knn_test[:min_len]

                perf_rf_fig = px.scatter(
                    x=y_test_plot, y=pred_rf_plot,
                    title="RandomForest: Predictions vs. Actual", labels={'x': 'Actual', 'y': 'Predicted'},
                    trendline="ols", trendline_color_override="red", opacity=0.7
                )
                perf_knn_fig = px.scatter(
                    x=y_test_plot, y=pred_knn_plot,
                    title="KNN: Predictions vs. Actual", labels={'x': 'Actual', 'y': 'Predicted'},
                    trendline="ols", trendline_color_override="red", opacity=0.7
                )
                scatter_plots = html.Div([
                    dcc.Graph(id='rf-pred-scatter', figure=perf_rf_fig),
                    dcc.Graph(id='knn-pred-scatter', figure=perf_knn_fig)
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
            except Exception as scatter_err:
                self.pipeline_logger.error(f"Failed to create scatter plots: {scatter_err}", exc_info=True)

            performance_summary = html.Div([
                html.Div([
                    html.H4("RandomForest Model"),
                    html.P(f"{metric_label}: {rf_metric:.4f}" if rf_metric is not None else "N/A")
                ], style={'width': '45%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px',
                          'margin': '10px', 'verticalAlign': 'top', 'textAlign': 'center'}),
                html.Div([
                    html.H4("KNN Model"),
                    html.P(f"{metric_label}: {knn_metric:.4f}" if knn_metric is not None else "N/A")
                ], style={'width': '45%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px',
                          'margin': '10px', 'verticalAlign': 'top', 'textAlign': 'center'})
            ])

            # Add bullet charts (if generated) to the children list
            perf_children.extend(bullet_charts)  # Add the bullet chart Div here

            perf_children.extend([
                performance_summary,
                scatter_plots,
                html.Div([
                    html.H4(plot_title),
                    html.Img(id='perf-main-plot', src=perf_img,
                             style={'maxWidth': '70%', 'height': 'auto', 'margin': 'auto',
                                    'display': 'block'}) if perf_img else html.P(
                        f"{plot_title} not generated or found.")
                ], style={'padding': '20px', 'textAlign': 'center', 'marginTop': '20px'})
            ])
        except Exception as perf_err:
            self.pipeline_logger.error(f"Error building performance tab: {perf_err}", exc_info=True)
            perf_children.append(html.P(f"Error displaying performance data: {perf_err}"))

        return html.Div(perf_children, style={'padding': '20px'})

    def _build_clustering_tab(self):
        self.pipeline_logger.debug("Building Clustering tab...")
        clustering_children = [html.H3("Clustering Results (Applied on Numeric Features)")]

        if self.processed_data is None or 'cluster' not in self.processed_data.columns:
            clustering_children.append(html.P(
                "Clustering results (e.g., 'cluster' column) not found in processed data. Clustering might have been skipped or failed."))
            return html.Div(clustering_children, style={'padding': '20px'})

        try:
            data_for_clustering_tab = self.processed_data.copy()
            data_for_clustering_tab['cluster'] = data_for_clustering_tab['cluster'].astype(
                str)  # Ensure categorical for color

            cluster_feature_options = [{'label': col, 'value': col} for col in data_for_clustering_tab.columns if
                                       col != 'cluster']
            numeric_cols_cluster = data_for_clustering_tab.select_dtypes(include=np.number).columns.tolist()
            default_x = numeric_cols_cluster[0] if len(numeric_cols_cluster) > 0 else cluster_feature_options[0][
                'value'] if cluster_feature_options else None
            default_y = numeric_cols_cluster[1] if len(numeric_cols_cluster) > 1 else cluster_feature_options[1][
                'value'] if len(cluster_feature_options) > 1 else default_x

            cluster_x_dropdown = dcc.Dropdown(id='cluster-x-selector', options=cluster_feature_options, value=default_x,
                                              clearable=False) if default_x else dcc.Dropdown(id='cluster-x-selector',
                                                                                              disabled=True,
                                                                                              placeholder="No columns available")
            cluster_y_dropdown = dcc.Dropdown(id='cluster-y-selector', options=cluster_feature_options, value=default_y,
                                              clearable=False) if default_y else dcc.Dropdown(id='cluster-y-selector',
                                                                                              disabled=True)

            clustering_children.extend([
                html.Div("Select features for scatter plot axes:"),
                html.Div([
                    html.Div([html.Label("X-axis:"), cluster_x_dropdown],
                             style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    html.Div([html.Label("Y-axis:"), cluster_y_dropdown],
                             style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'marginBottom': '15px'}),
                dcc.Graph(id='cluster-graph'),
                html.H3("Cluster Profiles (Mean Values of Numeric Features Used for Clustering)",
                        style={'marginTop': '20px'})
            ])

            if self.cluster_analysis_df is not None and not self.cluster_analysis_df.empty:
                cluster_table_data = self.cluster_analysis_df.reset_index().round(3).to_dict('records')
                cluster_table_cols = [{'name': col, 'id': col} for col in
                                      self.cluster_analysis_df.reset_index().columns]
                cluster_table = dash_table.DataTable(
                    id='cluster-profile-table', data=cluster_table_data, columns=cluster_table_cols,
                    page_size=10, style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left'},
                )
                clustering_children.append(cluster_table)
            else:
                clustering_children.append(html.P("Cluster profile analysis data not available."))
        except Exception as clust_err:
            self.pipeline_logger.error(f"Error building clustering tab: {clust_err}", exc_info=True)
            clustering_children.append(html.P(f"Error displaying clustering data: {clust_err}"))

        return html.Div(clustering_children, style={'padding': '20px'})

    def _build_predictions_tab(self):
        self.pipeline_logger.debug("Building Predictions tab...")
        predictions_children = [html.H3("Prediction Results (Sample from Test Set)")]

        if self.predictions is not None and not self.predictions.empty:
            predictions_sample = self.predictions.head(50)  # Show more samples
            # Round numeric columns for display
            numeric_pred_cols = predictions_sample.select_dtypes(include=np.number).columns
            predictions_sample.loc[:, numeric_pred_cols] = predictions_sample[numeric_pred_cols].round(3)

            predictions_children.append(dash_table.DataTable(
                id='predictions-table',
                data=predictions_sample.reset_index().to_dict('records'),  # Include index
                columns=[{'name': i, 'id': i} for i in predictions_sample.reset_index().columns],
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                sort_action='native'  # Enable sorting
            ))
        else:
            predictions_children.append(html.P("Predictions not available."))

        return html.Div(predictions_children, style={'padding': '20px'})

    def _build_shap_tab(self):
        self.pipeline_logger.debug("Building SHAP Importance tab...")
        shap_children = [html.H3("SHAP Feature Importance (Mean Absolute Value)")]

        if not shap_available:
            shap_children.append(html.P("SHAP library not installed."))
            return html.Div(shap_children, style={'padding': '20px'})
        if self.shap_values is None:
            shap_children.append(html.P("SHAP values not computed or available."))
            return html.Div(shap_children, style={'padding': '20px'})
        if not self.feature_names:
            shap_children.append(html.P("Feature names not available for SHAP plot."))
            return html.Div(shap_children, style={'padding': '20px'})

        try:
            shap_values_for_bar = self.shap_values
            # Handle multi-class: Average absolute SHAP across classes or use class 1? Let's average.
            if isinstance(self.shap_values, list):  # Multi-class list output
                if not self.shap_values:  # Empty list
                    raise ValueError("SHAP values list is empty.")
                abs_shap = [np.abs(s) for s in self.shap_values]
                # Check if all elements have same shape before averaging
                if len(set(s.shape for s in abs_shap)) > 1:
                    self.pipeline_logger.warning(
                        "SHAP values in list have different shapes. Using first element for bar chart.")
                    shap_importance = np.mean(abs_shap[0], axis=0)
                else:
                    # Ensure shapes are compatible before stacking and averaging
                    if len(set(s.shape[1] for s in abs_shap)) > 1:
                        raise ValueError("SHAP arrays in list have different number of features.")
                    shap_importance = np.mean(np.array(abs_shap), axis=(0, 2))  # Mean over samples and classes
            elif isinstance(self.shap_values,
                            np.ndarray) and self.shap_values.ndim == 3:  # Multi-class array output (samples, features, classes)
                shap_importance = np.mean(np.abs(self.shap_values), axis=(0, 2))  # Mean over samples and classes
            elif isinstance(self.shap_values,
                            np.ndarray) and self.shap_values.ndim == 2:  # Regression or single-class classification
                shap_importance = np.mean(np.abs(self.shap_values), axis=0)
            else:
                raise ValueError(f"Unexpected SHAP values format: {type(self.shap_values)}")

            # Ensure lengths match
            if len(shap_importance) != len(self.feature_names):
                raise ValueError(
                    f"SHAP importance length ({len(shap_importance)}) != feature names length ({len(self.feature_names)}).")

            shap_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': shap_importance}).sort_values(
                by='Importance', ascending=False).head(20)  # Top 20

            shap_fig_bar = px.bar(
                shap_df, x='Importance', y='Feature', orientation='h',
                title="Mean Absolute SHAP Value (Top 20 Features)"
            )
            shap_fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})  # Highest importance at top
            shap_children.append(dcc.Graph(id='shap-bar-chart', figure=shap_fig_bar))

        except Exception as shap_bar_err:
            self.pipeline_logger.error(f"Failed to create SHAP bar chart: {shap_bar_err}", exc_info=True)
            shap_children.append(html.P(f"Could not generate SHAP bar chart: {shap_bar_err}"))

        return html.Div(shap_children, style={'padding': '20px'})

    def _build_tree_tab(self, decision_tree_img):
        self.pipeline_logger.debug("Building Decision Tree tab...")
        img_style = {'maxWidth': '100%', 'height': 'auto', 'margin': 'auto', 'display': 'block'}
        return html.Div([
            html.H3("Example Decision Tree (RandomForest)"),
            html.P("Visualization of the first tree in the forest (max depth 3)."),
            html.Img(id='decision-tree-plot', src=decision_tree_img, style=img_style) if decision_tree_img else html.P(
                "Decision tree plot not generated or found."),
        ], style={'padding': '20px'})

    def _define_dashboard_callbacks(self, app):
        """ Defines the callbacks for the Dash application. """
        self.pipeline_logger.info("Defining dashboard callbacks...")

        @app.callback(
            [Output('histogram', 'figure'),
             Output('stats-table', 'children')],
            [Input('column-selector', 'value')],
            prevent_initial_call=True
        )
        def update_overview_visualizations(selected_col):
            """Updates histogram and stats table based on selected column."""
            if selected_col is None or self.processed_data is None:
                return px.histogram(title='Select a column'), html.Div("No data to display stats for.")

            fig = go.Figure()
            stats_table_div = html.Div("Could not generate statistics.")  # Default error message

            try:
                data_col = self.processed_data[selected_col]
                col_dtype = data_col.dtype

                # Create plot based on dtype
                if pd.api.types.is_numeric_dtype(col_dtype):
                    fig = px.histogram(self.processed_data, x=selected_col, nbins=50,
                                       title=f"Distribution of {selected_col}")
                elif isinstance(col_dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
                        col_dtype) or pd.api.types.is_bool_dtype(col_dtype):
                    # For categorical/object/bool, show top N categories if too many
                    top_n = 25
                    counts = data_col.value_counts()
                    title = f"Frequency of {selected_col}"
                    if len(counts) > top_n:
                        top_counts = counts.nlargest(top_n)
                        fig = px.bar(top_counts, x=top_counts.index.astype(str), y=top_counts.values,
                                     title=f"{title} (Top {top_n})")
                        fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                    else:
                        fig = px.bar(counts, x=counts.index.astype(str), y=counts.values, title=title)
                        fig.update_layout(xaxis_title=selected_col, yaxis_title="Count")
                else:
                    fig = go.Figure(layout={'title': f"Cannot plot distribution for dtype: {col_dtype}"})

                # Generate stats table
                stats = data_col.describe().to_frame().reset_index()
                stats.columns = ['Statistic', 'Value']
                # Format numeric values in stats
                stats['Value'] = stats['Value'].apply(
                    lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x)

                stats_table_div = dash_table.DataTable(
                    id='desc-stats-table',
                    columns=[{'name': 'Statistic', 'id': 'Statistic'}, {'name': 'Value', 'id': 'Value'}],
                    data=stats.to_dict('records'),
                    style_table={'overflowX': 'auto', 'width': '50%'},  # Adjust width
                    style_cell={'textAlign': 'left'},
                )
            except Exception as e:
                self.pipeline_logger.error(f"Error updating overview plot/stats for {selected_col}: {e}", exc_info=True)
                fig = go.Figure(layout={'title': f'Error loading data for {selected_col}'})
                stats_table_div = html.P(f"Could not generate plot/statistics for {selected_col}: {e}")

            return fig, stats_table_div

        @app.callback(
            Output('cluster-graph', 'figure'),
            [Input('cluster-x-selector', 'value'),
             Input('cluster-y-selector', 'value')],
            prevent_initial_call=True
        )
        def update_cluster_scatter(selected_x, selected_y):
            """Updates the cluster scatter plot based on selected axes."""
            fig = go.Figure()  # Default empty figure
            if self.processed_data is None or 'cluster' not in self.processed_data.columns:
                return fig.update_layout(title="Clustering data not available")
            if not selected_x or not selected_y:
                return fig.update_layout(title="Select X and Y axes for cluster plot")

            try:
                # Use processed_data which has the cluster column
                data_for_plot = self.processed_data.copy()
                # Ensure cluster is string for discrete color mapping
                data_for_plot['cluster'] = data_for_plot['cluster'].astype(str)

                fig = px.scatter(
                    data_for_plot,
                    x=selected_x,
                    y=selected_y,
                    color='cluster',  # Color by the cluster column
                    title=f"Cluster Visualization: {selected_x} vs {selected_y}",
                    hover_data=data_for_plot.columns,  # Show all data on hover
                    opacity=0.7
                )
                fig.update_layout(xaxis_title=selected_x, yaxis_title=selected_y, legend_title_text='Cluster')
            except Exception as e:
                self.pipeline_logger.error(f"Error updating cluster plot for {selected_x} vs {selected_y}: {e}",
                                           exc_info=True)
                fig = go.Figure().update_layout(title=f"Error loading cluster plot: {e}")

            return fig

        self.pipeline_logger.info("Dashboard callbacks defined.")

    # =================== Scheduling & Pipeline ===================

    def schedule_pipeline(self, task_type=None):
        """ Allows user to schedule runs of the pipeline with daily, weekly, or monthly frequency. """
        try:
            print("\n--- Scheduling Configuration ---")
            print("Pipeline scheduling is recommended to keep your models and insights up-to-date.")

            # --- Frequency Selection ---
            while True:
                print("\nSelect scheduling frequency:")
                print("1. Daily")
                print("2. Weekly")
                print("3. Monthly")
                schedule_frequency_choice = input("Enter your choice (1/2/3): ").strip()

                if schedule_frequency_choice == '1':
                    schedule_frequency = 'daily'
                    break
                elif schedule_frequency_choice == '2':
                    schedule_frequency = 'weekly'
                    break
                elif schedule_frequency_choice == '3':
                    schedule_frequency = 'monthly'
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

            # --- Time Input ---
            while True:
                schedule_time = input(
                    f"Enter time for {schedule_frequency} run (HH:MM, 24-hr format, e.g., 08:30 or 23:00): ").strip()
                try:
                    # Validate time format
                    time.strptime(schedule_time, '%H:%M')
                    break  # Valid time format
                except ValueError:
                    print("Invalid time format. Please use HH:MM (e.g., 14:00).")

            # --- Define the core job ---
            def job():
                self.scheduler_logger.info(f"--- Scheduled {schedule_frequency.upper()} Pipeline Run Starting ---")
                try:
                    # Re-run the pipeline: reload data, re-process, re-train
                    # Don't start a *new* dashboard each time
                    self.run_pipeline(start_dashboard=False, task_type=task_type,
                                      reload_data=True)  # Reload data for scheduled runs
                    self.scheduler_logger.info(
                        f"--- Scheduled {schedule_frequency.upper()} Pipeline Run Completed Successfully ---")
                except Exception as e:
                    self.scheduler_logger.error(
                        f"--- Scheduled {schedule_frequency.upper()} Pipeline Run Failed: {e} ---", exc_info=True)

            # --- Schedule based on frequency ---
            scheduled_message = ""
            if schedule_frequency == 'daily':
                schedule.every().day.at(schedule_time).do(job)
                scheduled_message = f"Pipeline scheduled to run DAILY at {schedule_time}."

            elif schedule_frequency == 'weekly':
                days_of_week = {
                    "monday": schedule.every().monday,
                    "tuesday": schedule.every().tuesday,
                    "wednesday": schedule.every().wednesday,
                    "thursday": schedule.every().thursday,
                    "friday": schedule.every().friday,
                    "saturday": schedule.every().saturday,
                    "sunday": schedule.every().sunday
                }
                while True:
                    schedule_day_of_week = input(
                        f"Enter day of the week for weekly run ({', '.join(days_of_week.keys())}): ").strip().lower()
                    if schedule_day_of_week in days_of_week:
                        days_of_week[schedule_day_of_week].at(schedule_time).do(job)
                        scheduled_message = f"Pipeline scheduled to run WEEKLY every {schedule_day_of_week.capitalize()} at {schedule_time}."
                        break
                    else:
                        print(f"Invalid day. Please enter one of: {', '.join(days_of_week.keys())}")

            elif schedule_frequency == 'monthly':
                while True:
                    try:
                        schedule_day_of_month = int(input("Enter day of the month for monthly run (1-31): ").strip())
                        if 1 <= schedule_day_of_month <= 31:
                            break
                        else:
                            print("Invalid day. Please enter a number between 1 and 31.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                # Workaround for monthly scheduling with the 'schedule' library:
                # Schedule a daily check and run the job only on the target day.
                def monthly_job_wrapper():
                    today = datetime.date.today()
                    if today.day == schedule_day_of_month:
                        self.scheduler_logger.info(f"Today is day {schedule_day_of_month}, executing monthly job.")
                        job()  # Execute the actual pipeline job
                    else:
                        self.scheduler_logger.debug(
                            f"Skipping monthly job. Today is day {today.day}, target is {schedule_day_of_month}.")

                schedule.every().day.at(schedule_time).do(monthly_job_wrapper)
                scheduled_message = f"Pipeline scheduled to run MONTHLY on day {schedule_day_of_month} at {schedule_time}."
                print(
                    f"(Note: For monthly runs on day {schedule_day_of_month}, the job will not run in months without that day, e.g., day 31 in February).")

            # --- Log and start scheduler loop ---
            self.scheduler_logger.info(scheduled_message)
            print(f"\n{scheduled_message}")
            print("Scheduler is running in the background. Press Ctrl+C to stop the scheduler.")

            # Keep the script running to allow the scheduler to work
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            self.scheduler_logger.info("Scheduler stopped by user (Ctrl+C).")
            print("\nScheduler stopped.")
        except Exception as e:
            self.scheduler_logger.error(f"An error occurred in the scheduling setup: {e}", exc_info=True)
            print(f"\nError during scheduling setup: {e}")
            # Don't raise here, allow main program flow to potentially continue or exit gracefully

    def run_pipeline(self, start_dashboard=True, task_type=None, reload_data=True, is_initial_run=False):
        """
        Main pipeline execution flow: Load, Process, Train/Predict, Dashboard.

        Args:
            start_dashboard (bool): Whether to start the Dash dashboard.
            task_type (str, optional): 'classification' or 'regression'. Defaults to None (auto-detect).
            reload_data (bool): Whether to attempt reloading data from the source.
            is_initial_run (bool): Flag to indicate if this is the very first execution.
        """
        start_time = time.time()
        self.pipeline_logger.info("=" * 30 + " Pipeline Execution Started " + "=" * 30)
        try:
            # --- Data Ingestion ---
            # Determine if we need to load/reload data
            load_needed = False
            if reload_data:
                # If reload is requested, we always need to load something
                load_needed = True
            elif self.original_data is None:
                # If not reloading, but no data exists yet, we need to load
                load_needed = True
                self.pipeline_logger.info("No data loaded previously and reload=False. Triggering initial load.")
            else:
                # Not reloading and data exists, use existing data
                self.pipeline_logger.info("Using previously loaded data (not reloading from source).")
                # Ensure self.data is reset from original for processing
                self.data = self.original_data.copy()

            if load_needed:
                # Check if data source parameters are stored (meaning it's not the very first run)
                can_auto_reload = self.data_source_type and (self.file_path or self.db_connection_string)

                if can_auto_reload and not is_initial_run:
                    # --- Automatic Reload Path (Scheduled Runs or subsequent manual runs with reload=True) ---
                    self.pipeline_logger.info("Attempting automatic data reload...")
                    try:
                        self._reload_data_automatically()
                    except Exception as auto_reload_err:
                        # If auto-reload fails, log error and stop this pipeline run
                        self.pipeline_logger.error(
                            f"Stopping pipeline run due to automatic data reload failure: {auto_reload_err}",
                            exc_info=True)
                        print(f"\nERROR: Pipeline run failed during automatic data reload. Check logs.")
                        return  # Exit the run_pipeline method
                else:
                    # --- Interactive Load Path (Very First Run or if auto-reload isn't possible) ---
                    if is_initial_run:
                        self.pipeline_logger.info("Initial run: Triggering interactive data source selection.")
                        # No warning message here for the first run
                    else:
                        # This case might happen if parameters were somehow lost or reload=True was forced without params
                        self.pipeline_logger.warning(
                            "Data source parameters not found or initial run flag not set. Triggering interactive prompt.")
                    self.prompt_for_dataset()  # Needs interactive setup

            # Check if data loading was successful before proceeding
            if self.data is None or self.data.empty:
                self.pipeline_logger.error("Pipeline cannot proceed: Data is empty or failed to load.")
                print("\nERROR: Pipeline cannot proceed as data is missing or empty.")
                return  # Exit the run_pipeline method

            # --- Data Processing ---
            # process_data now correctly handles interactive vs non-interactive target selection based on sys.stdin.isatty()
            # and prioritizes self.target_col if it exists and is valid.
            self.pipeline_logger.info("Starting data processing...")
            # Pass the is_initial_run flag to process_data
            self.process_data(manual_target_col=None, is_initial_run=is_initial_run)  # Pass the flag here
            self.pipeline_logger.info("Data processing finished.")

            # --- Training & Prediction ---
            self.pipeline_logger.info("Starting model training and prediction...")
            # Pass the user-specified or auto-detected task_type
            # train_predict uses self.task_type if task_type is None
            # Ensure self.task_type is updated if auto-detected during train_predict
            if task_type:
                self.task_type = task_type  # Use provided task type
            self.train_predict(task_type=self.task_type)  # Pass the potentially updated self.task_type
            self.pipeline_logger.info("Model training and prediction finished.")

            # --- Dashboard ---
            if start_dashboard:
                self.pipeline_logger.info("Starting dashboard creation...")
                self.create_dashboard()  # Runs server in a separate thread
                self.pipeline_logger.info("Dashboard creation process initiated.")
                # Main thread continues after starting dashboard thread
            else:
                self.pipeline_logger.info("Skipping dashboard creation as requested.")

            end_time = time.time()
            total_duration = end_time - start_time
            self.pipeline_logger.info(f"--- Pipeline executed successfully in {total_duration:.2f} seconds ---")
            print("\n--- Pipeline executed successfully ---")
            print(f"Output files (plots, processed data, metrics, logs) are in: {self.visualization_dir}")
            if start_dashboard:
                print("Dashboard should be running (check logs/browser).")


        except FileNotFoundError as e:
            self.pipeline_logger.error(f"Pipeline failed: Data file not found. {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed - Data file not found: {e}")
        except ValueError as e:
            self.pipeline_logger.error(f"Pipeline failed: Invalid input or data issue. {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed - {e}")
        except MemoryError as e:
            self.pipeline_logger.error(f"Pipeline failed: Out of memory. {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed - Out of memory. Try with a smaller dataset or simpler model.")
        except Exception as e:
            self.pipeline_logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
            print(f"\nERROR: Pipeline failed unexpectedly. Check 'pipeline.log' for details: {e}")
            # Optionally re-raise for critical failures if needed for external handling
            # raise

        finally:
            self.pipeline_logger.info("=" * 30 + " Pipeline Execution Finished " + "=" * 30)

# =================== Main Execution Block ===================
if __name__ == "__main__":

    automation = None  # Initialize to None
    try:
        automation = DataScienceAutomation()
        # Setup signal handler for graceful exit (especially for scheduler)
        signal.signal(signal.SIGINT, automation.graceful_exit)
        signal.signal(signal.SIGTERM, automation.graceful_exit)  # Handle termination signal too

        print("\n--- Automated Data Science Pipeline ---")
        print("Starting initial pipeline run and dashboard creation...")

        # Ask for task type upfront (optional) - This remains as it's configuration for the run
        task_type_input = input(
            "Enter task type (classification/regression) or press Enter for auto-detection: ").strip().lower()
        task_type = task_type_input if task_type_input in ['classification', 'regression'] else None
        if task_type:
            automation.pipeline_logger.info(f"User specified task type: {task_type}")
            print(f"Task type set to: {task_type}")
        else:
            automation.pipeline_logger.info("Task type will be auto-detected.")
            print("Task type will be auto-detected.")

        # --- Run the pipeline once and launch the dashboard ---
        # Pass the user-specified task_type (or None) to run_pipeline
        # reload_data=True ensures data is loaded on this first run.
        # **** ADD is_initial_run=True HERE ****
        automation.run_pipeline(start_dashboard=True, task_type=task_type, reload_data=True, is_initial_run=True)

        # --- Proceed to Scheduling Configuration AFTER successful initial run/dashboard launch ---
        if automation.dashboard_running:
            print("\nInitial run complete. Dashboard is running in the background.")
            print("Proceeding to configure the schedule for future automated runs.")

            # Pass the determined task_type (either user-provided or auto-detected during the first run)
            # to schedule_pipeline so scheduled jobs use the correct type.
            # The schedule_pipeline method itself handles the user interaction for setting up the schedule.
            # **** Pass the potentially updated automation.task_type ****
            automation.schedule_pipeline(task_type=automation.task_type)

            # --- Keep the main thread alive for the scheduler and dashboard ---
            # The schedule_pipeline method now contains the loop that checks schedule.run_pending()
            # However, the main script still needs to stay alive for the dashboard thread
            # and the scheduler loop within schedule_pipeline to function.
            print("\nScheduling configured. The scheduler is now active.")
            print("IMPORTANT: Keep this terminal window open for the scheduler and dashboard to remain active.")
            print("Press Ctrl+C in this terminal to stop the entire program (scheduler and dashboard).")
            try:
                # Keep main thread alive indefinitely until interrupted
                while True:
                    time.sleep(1)  # Minimal sleep to prevent busy-waiting
            except KeyboardInterrupt:
                # This will be caught here if Ctrl+C is pressed *after* scheduling is set up.
                # The graceful_exit handler should ideally manage the shutdown.
                print("\nCtrl+C detected during active run. Exiting program.")
                automation.pipeline_logger.info("Exiting program after scheduling was active (Ctrl+C).")
                # sys.exit(0) # graceful_exit handles this

        else:
            # This block executes if the dashboard failed to start after the initial run.
            print("\nPipeline finished, but the dashboard did not start or failed (check logs).")
            print("Scheduling will be skipped as the dashboard is not running.")
            automation.pipeline_logger.warning("Dashboard failed to start. Skipping scheduling configuration.")


    except KeyboardInterrupt:
        # This catches Ctrl+C if pressed *during* the initial setup/run/scheduling config
        print("\nOperation cancelled by user during setup or initial run.")
        if automation:
            automation.pipeline_logger.warning("Pipeline setup/initial run interrupted by user (KeyboardInterrupt).")
        else:
            logger.warning("Pipeline setup interrupted by user before initialization.")
    except (FileNotFoundError, ValueError, MemoryError) as e:
        # These errors are logged within run_pipeline, just print final message
        print(f"\nCritical Error during pipeline execution. Check 'pipeline.log'. Exiting.")
        # No sys.exit needed here, program will terminate naturally after the exception.
    except Exception as e:
        # Catch-all for unexpected errors during setup or if run_pipeline raised it
        error_msg = f"A critical error occurred in the main execution block: {e}"
        print(f"\n{error_msg}")
        if automation and hasattr(automation, 'pipeline_logger'):
            automation.pipeline_logger.critical(error_msg, exc_info=True)
        else:
            logger.critical(error_msg, exc_info=True)  # Use base logger
        sys.exit(1)  # Exit with error code for unexpected critical errors

    finally:
        logging.shutdown()  # Ensure all logs are flushed before exiting
        print("\nProgram finished.")
