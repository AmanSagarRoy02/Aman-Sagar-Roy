import os
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, dash_table
import json
import schedule
import time
import signal
import sys
from calendar import monthrange
import logging

# Configure logging
logging.basicConfig(
    filename="pipeline.log",  # Log file
    level=logging.INFO,       # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)
logger = logging.getLogger()

# Log to both console and file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logging.basicConfig(
        filename="scheduler.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def graceful_exit(signal_received, frame):
    print("Scheduler stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)

def log_event(message):
    logging.info(message)
    print(message)


class DataScienceAutomation:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.original_data = None
        self.model = None
        self.predictions = None
        self.label_encoders = {}
        self.visualization_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.visualization_dir, exist_ok=True)
        logger.info("DataScienceAutomation initialized.")

    def preview(self, message, df, n=5):
        """Preview a dataframe or message."""
        print(f"\n{message}\n")
        print(df.head(n) if isinstance(df, pd.DataFrame) else df)

    def prompt_for_dataset(self):
        try:
            print("Enter the data source type ('file' or 'database'):")
            source_type = input("Source Type: ").strip().lower()

            if source_type == 'file':
                self.load_file()
            elif source_type == 'database':
                self.load_database()
            else:
                raise ValueError("Unsupported source type. Please enter 'file' or 'database'.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def load_file(self):
        try:
            print("Enter the file path for your dataset:")
            file_path = input("File Path: ").strip().strip('"').strip("'")
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == '.csv':
                self.data = pd.read_csv(file_path, delimiter=',' , encoding='ISO-8859-1')
                logger.info(f"Loaded dataset from CSV file: {file_path}")
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
                logger.info(f"Loaded dataset from Excel file: {file_path}")
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
                logger.info(f"Loaded dataset from JSON file: {file_path}")
            elif file_extension == '.parquet':
                self.data = pd.read_parquet(file_path)
                logger.info(f"Loaded dataset from Parquet file: {file_path}")
            elif file_extension == '.txt':
                delimiter = input("Enter delimiter for the text file (e.g., ',' for CSV): ").strip()
                self.data = pd.read_csv(file_path, delimiter=delimiter ,  encoding='ISO-8859-1')
                logger.info(f"Loaded dataset from text file with delimiter '{delimiter}': {file_path}")
            else:
                raise ValueError(f"Unsupported file format '{file_extension}'. Supported formats are: CSV, Excel, JSON, Parquet, TXT.")

            self.original_data = self.data.copy()
            logger.info(f"Dataset loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading file data: {e}")
            raise

    def load_database(self):
        try:
            print("Enter the database type ('sqlite' or 'sqlalchemy'):")
            db_type = input("Database Type: ").strip().lower()

            if db_type == 'sqlite':
                print("Enter the SQLite database file path:")
                db_path = input("SQLite File Path: ").strip().strip('"').strip("'")
                conn = sqlite3.connect(db_path)
                print("Enter the SQL query to fetch data:")
                query = input("SQL Query: ").strip()
                self.data = pd.read_sql_query(query, conn)
                conn.close()
            elif db_type == 'sqlalchemy':
                print("Enter the SQLAlchemy connection string (e.g., 'postgresql://user:password@host/dbname'):")
                conn_string = input("Connection String: ").strip()
                engine = create_engine(conn_string)
                print("Enter the SQL query to fetch data:")
                query = input("SQL Query: ").strip()
                self.data = pd.read_sql_query(query, engine)
            else:
                raise ValueError("Unsupported database type. Please enter 'sqlite' or 'sqlalchemy'.")

            self.original_data = self.data.copy()
            self.preview("Data loaded successfully (Preview):", self.data)
            logger.info(f"Dataset columns: {self.data.columns}")
            logger.info(f"Dataset shape after loading: {self.data.shape}")

        except Exception as e:
            print(f"Error loading file data: {e}")
            raise

    def select_target_column(self):
        """Select the target column dynamically by checking from the last column backward."""
        for col in reversed(self.data.columns):
            if self.data[col].notnull().any():  # Check if column has at least one non-null value
                logger.info(f"Selected target column: '{col}'")
                return col
            else:
                logger.warning(f"Column '{col}' is completely empty. Skipping.")
        raise ValueError("No valid column found to set as the target column.")

    def process_data(self):
        try:
            logger.info("Processing data...")
            self.data.drop_duplicates(inplace=True)
            logger.info(f"Dropped duplicate rows. Remaining rows: {self.data.shape[0]}")

            for col in self.data.columns:
                if self.data[col].apply(lambda x: isinstance(x, str) and str(x).startswith("{")).any():
                    self.data[col] = self.data[col].apply(lambda x: self.parse_json_safe(x))
                    logger.info(f"Parsed JSON-like data in column '{col}'.")

            # Drop completely missing columns
            for col in self.data.columns:
                if self.data[col].isnull().mean() == 1.0:
                    logger.warning(f"Column '{col}' is completely missing. Dropping column.")
                    self.data.drop(columns=[col], inplace=True)

            # Dynamically select the target column
            target_col = self.select_target_column()

            # Encode the target column if it is categorical
            if self.data[target_col].dtype == 'object':
                logger.info(f"Encoding target column '{target_col}' using LabelEncoder.")
                le = LabelEncoder()
                self.data[target_col] = le.fit_transform(self.data[target_col])
                self.label_encoders[target_col] = le

            # Handle missing values in non-target columns
            for col in self.data.columns:
                if col != target_col and self.data[col].isnull().any():
                    if self.data[col].dtype in ['float64', 'int64']:
                        self.data[col].fillna(self.data[col].mean(), inplace=True)
                        logger.info(f"Filled missing values in numeric column '{col}' with mean.")
                    elif self.data[col].dtype == 'object':
                        self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                        logger.info(f"Filled missing values in categorical column '{col}' with mode.")

            # Handle missing values in the target column
            if self.data[target_col].isnull().any():
                missing_ratio = self.data[target_col].isnull().mean()
                logger.warning(
                    f"Target column '{target_col}' contains missing values. Missing ratio: {missing_ratio:.2%}")

                if missing_ratio < 0.2:  # Impute values if less than 20% missing
                    if self.data[target_col].dtype in ['float64', 'int64']:  # Numeric target
                        self.data.loc[:, target_col] = self.data[target_col].fillna(self.data[target_col].mean())
                        logger.info(f"Imputed missing values in numeric target column '{target_col}' using mean.")
                    else:  # Categorical target
                        self.data.loc[:, target_col] = self.data[target_col].fillna(self.data[target_col].mode()[0])
                        logger.info(f"Imputed missing values in categorical target column '{target_col}' using mode.")
                else:  # Drop column or abort based on missing threshold
                    logger.error(
                        f"Target column '{target_col}' has too many missing values ({missing_ratio:.2%}). Aborting.")
                    raise ValueError(
                        f"Too many missing values in target column '{target_col}' ({missing_ratio:.2%}).")

            # Handle datetime columns
            for col in self.data.columns:
                if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                    self.data[f'{col}_year'] = self.data[col].dt.year
                    self.data[f'{col}_month'] = self.data[col].dt.month
                    self.data[f'{col}_day'] = self.data[col].dt.day
                    self.data[f'{col}_dayofweek'] = self.data[col].dt.dayofweek
                    self.data.drop(columns=[col], inplace=True)
                    logger.info(f"Extracted year, month, day, and dayofweek from column '{col}'.")

            logger.info(f"Column Data Types:\n{self.data.dtypes}")

            # Encode only necessary categorical columns
            for col in self.data.select_dtypes(include=['object', 'bool']).columns:
                if col != target_col:  # Skip encoding the target column again
                    logger.info(f"Encoding categorical column '{col}' using LabelEncoder.")
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.label_encoders[col] = le

            # Validate dataset
            if self.data.empty:
                raise ValueError("Dataset is empty after processing. Ensure valid input data.")
            if self.data[target_col].isnull().any():
                raise ValueError(f"Target column '{target_col}' contains missing values after processing.")
            logger.info("Data processing complete.")


            # Save the processed dataset
            processed_file_path = os.path.join(self.visualization_dir, "processed_dataset.parquet")
            self.data.to_parquet(processed_file_path, index=False)
            self.processed_data = self.data.copy()
            logger.info(f"Processed dataset saved to {processed_file_path}")
            # Return the selected target column for further processing
            return target_col

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def parse_json_safe(self, value):
        """
        Safely parses a JSON-like string and extracts the 'name' field if present.
        """
        if isinstance(value, str) and value.startswith("{"):
            try:
                formatted_value = value.replace("'", '"')
                return json.loads(formatted_value).get('name', None)
            except json.JSONDecodeError:
                return None
        return value

    def impute_missing_target(self, target_col):
        missing_target_rows = self.data[self.data[target_col].isnull()]
        available_target_rows = self.data[self.data[target_col].notnull()]
        if missing_target_rows.empty:
            return
        if self.data[target_col].dtype in ['float64', 'int64']:
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        X_train = available_target_rows.drop(columns=[target_col])
        y_train = available_target_rows[target_col]
        X_missing = missing_target_rows.drop(columns=[target_col])

        X_train, X_missing = self.encode_features(X_train, X_missing)
        model.fit(X_train, y_train)
        self.data.loc[missing_target_rows.index, target_col] = model.predict(X_missing)

    def encode_features(self, train_data, test_data):
        """Encodes non-numeric features in train and test datasets."""
        encoders = {}
        for col in train_data.columns:
            if train_data[col].dtype == 'object':
                le = LabelEncoder()
                train_data[col] = le.fit_transform(train_data[col].astype(str))
                test_data[col] = test_data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                encoders[col] = le
        return train_data, test_data

    def train_predict(self):
        try:
            logger.info("Training and predicting...")

            # Validate dataset structure
            if self.data.shape[1] < 2:
                raise ValueError("Dataset must have at least two columns (features and target).")

            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]

            # Ensure X and y are not empty
            if X.empty or y.empty:
                raise ValueError("Features (X) or target (y) are empty after processing.")

            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

            if y.isna().any():
                raise ValueError("Target variable contains missing values. Handle them before training.")

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if y.nunique() <= 20 and y.dtype in ['int64', 'object']:
                self.model = RandomForestClassifier(random_state=42)
                logger.info("Detected Classification Task. Using RandomForestClassifier.")
            else:
                self.model = RandomForestRegressor(random_state=42)
                logger.info("Detected Regression Task. Using RandomForestRegressor.")

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            logger.info("Model training completed. Predictions generated.")

            self.generate_insights(y_test, y_pred)
            self.predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            predictions_file_path = os.path.join(self.visualization_dir, "predictions.csv")
            self.predictions.to_csv(predictions_file_path, index=False)
            logger.info(f"Predictions saved to {predictions_file_path}")
        except Exception as e:
            logger.error(f"Error in training/prediction: {e}")
            raise

    def generate_insights(self, y_test, y_pred):
        try:
            print("Generating insights from predictions...")  # Debugging output

            if isinstance(self.model, RandomForestClassifier):
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                print(f"Accuracy: {accuracy:.2f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
                sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title("Confusion Matrix")
                plt.savefig(os.path.join(self.visualization_dir, "confusion_matrix.png"))
                plt.show()

                insights = {"Accuracy": accuracy, "Classification Report": report}

            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Mean Squared Error (MSE): {mse:.2f}")
                print(f"R-squared (R2): {r2:.2f}")

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Predictions vs Actual")
                plt.savefig(os.path.join(self.visualization_dir, "predictions_vs_actual.png"))
                plt.show()

                insights = {"MSE": mse, "R2 Score": r2}

            insights_file_path = os.path.join(self.visualization_dir, "insights.json")
            with open(insights_file_path, 'w') as f:
                json.dump(insights, f, indent=4)

            print(f"Insights saved to {insights_file_path}")
        except Exception as e:
            print(f"Error generating insights: {e}")
            raise

    def create_dashboard(self, debug=False):
        try:
            logger.info("Creating dashboard...")
            app = Dash(__name__)

            # 1. Correlation Heatmap for Numeric Features
            corr_matrix = self.processed_data.select_dtypes(include=['float64', 'int64']).corr()
            heatmap_fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Feature Correlation Heatmap",
                labels={"color": "Correlation Coefficient"},
                template="plotly_dark"
            )

            # 2. Predictions vs Actual Scatter Plot (Regression)
            scatter_fig = px.scatter(
                self.predictions, x="Actual", y="Predicted",
                labels={"Actual": "True Value", "Predicted": "Predicted Value"},
                title="Predictions vs Actual",
                template="plotly_dark"
            )
            scatter_fig.add_shape(
                type="line",
                line=dict(dash="dash", color="red"),
                x0=min(self.predictions["Actual"]),
                x1=max(self.predictions["Actual"]),
                y0=min(self.predictions["Actual"]),
                y1=max(self.predictions["Actual"])
            )

            # 3. Feature Importance (if available)
            importance_fig = None
            if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
                importances = self.model.feature_importances_
                feature_names = self.processed_data.columns[:-1]
                importance_fig = px.bar(
                    x=feature_names,
                    y=importances,
                    labels={"x": "Feature", "y": "Importance Score"},
                    title="Feature Importance",
                    template="plotly_dark"
                )

            # 4. Histograms for Numeric Columns
            histograms = []
            numeric_cols = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_cols:
                hist_fig = px.histogram(
                    self.processed_data,
                    x=column,
                    title=f"Distribution of {column}",
                    labels={column: column, "count": "Frequency"},
                    template="plotly_dark"
                )
                histograms.append(hist_fig)

            # 5. Bar Plots for Categorical Columns
            bar_plots = []
            categorical_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
            for column in categorical_cols:
                bar_fig = px.bar(
                    self.processed_data[column].value_counts().reset_index(),
                    x="index",
                    y=column,
                    title=f"Frequency of {column}",
                    labels={"index": column, column: "Count"},
                    template="plotly_dark"
                )
                bar_plots.append(bar_fig)

            # Layout
            app.layout = html.Div([
                html.H1("Data Science Automation Dashboard", style={'textAlign': 'center'}),
                html.Div("Welcome to the Data Science Automation Dashboard. Explore insights below."),

                # Model Predictions
                html.H2("Model Predictions", style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='predictions-table',
                    columns=[{"name": col, "id": col} for col in self.predictions.columns],
                    data=self.predictions.to_dict('records'),
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'center'}
                ),
                dcc.Graph(figure=scatter_fig),

                # Feature Correlations
                html.H2("Feature Correlations", style={'textAlign': 'center'}),
                dcc.Graph(figure=heatmap_fig),

                # Feature Importance
                html.H2("Feature Importance", style={'textAlign': 'center'}),
                dcc.Graph(figure=importance_fig if importance_fig else go.Figure()),

                # Data Distributions
                html.H2("Data Distributions", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(figure=hist, style={'display': 'inline-block', 'width': '48%'})
                    for hist in histograms
                ], style={'textAlign': 'center', 'flexWrap': 'wrap', 'display': 'flex'}),

                # Categorical Feature Frequencies
                html.H2("Categorical Feature Frequencies", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(figure=bar, style={'display': 'inline-block', 'width': '48%'})
                    for bar in bar_plots
                ], style={'textAlign': 'center', 'flexWrap': 'wrap', 'display': 'flex'}),
            ])

            logger.info("Starting the dashboard server...")
            app.run_server(debug=debug, port=8050)
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise

    def configure_schedule(self):
        schedule_config = {
            "enabled": True,  # Enable/disable scheduling
            "frequency": "monthly",  # Options: daily, weekly, monthly, yearly
            "time": "12:00",  # Time in HH:MM 24-hour format
            "day_of_week": "Monday",  # For weekly schedule
            "day_of_month": 15,  # For monthly schedule
            "date": "12-25",  # For yearly schedule
        }
        return schedule_config

    def schedule_pipeline(self):
        try:
            config = self.configure_schedule()

            if not config.get("enabled", False):
                log_event("Pipeline scheduling is disabled.")
                return

            frequency = config["frequency"].lower()
            run_time = config["time"]

            try:
                time.strptime(run_time, "%H:%M")
            except ValueError:
                raise ValueError("Invalid time format in configuration. Use HH:MM 24-hour format.")

            if frequency == "daily":
                log_event(f"Scheduling pipeline to run daily at {run_time}...")
                schedule.every().day.at(run_time).do(self.run_pipeline)

            elif frequency == "weekly":
                day_of_week = config["day_of_week"]
                if day_of_week.capitalize() not in schedule.WEEKDAY_NAMES:
                    raise ValueError("Invalid day of the week in configuration.")
                log_event(f"Scheduling pipeline to run weekly on {day_of_week} at {run_time}...")
                schedule.every().week.at(run_time).do(self.run_pipeline)

            elif frequency == "monthly":
                day_of_month = config["day_of_month"]
                if not isinstance(day_of_month, int) or not (1 <= day_of_month <= 31):
                    raise ValueError("Invalid day of the month in configuration.")
                log_event(f"Scheduling pipeline to run monthly on day {day_of_month} at {run_time}...")

                def check_monthly():
                    now = time.localtime()
                    last_day = monthrange(now.tm_year, now.tm_mon)[1]
                    if now.tm_mday == min(day_of_month, last_day):
                        self.run_pipeline()

                schedule.every().day.at(run_time).do(check_monthly)

            elif frequency == "yearly":
                date = config["date"]
                try:
                    month, day = map(int, date.split("-"))
                    time.strptime(f"{month:02d}-{day:02d}", "%m-%d")
                    log_event(f"Scheduling pipeline to run yearly on {date} at {run_time}...")

                    def check_yearly():
                        now = time.localtime()
                        if now.tm_mon == month and now.tm_mday == day:
                            self.run_pipeline()

                    schedule.every().day.at(run_time).do(check_yearly)
                except ValueError:
                    raise ValueError("Invalid date format in configuration (use MM-DD).")

            else:
                raise ValueError("Invalid frequency in configuration. Use 'daily', 'weekly', 'monthly', or 'yearly'.")

            log_event("Scheduler started.")
            while True:
                schedule.run_pending()
                time.sleep(1)

        except Exception as e:
            log_event(f"Error in scheduling pipeline: {e}")
            raise

    def run_pipeline(self):
        print("Starting pipeline...")
        try:
            self.prompt_for_dataset()
            self.process_data()
            self.train_predict()
            self.create_dashboard()
        except Exception as e:
            print(f"Pipeline failed: {e}")
            log_event(f"Pipeline failed: {e}")

if __name__ == "__main__":
    automation = DataScienceAutomation()
    automation.run_pipeline()
    automation.schedule_pipeline()