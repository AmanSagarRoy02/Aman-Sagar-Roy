Data Science Automation Pipeline
This project provides a Data Science Automation Pipeline that simplifies the process of data preprocessing, model training, prediction, and insights generation. It supports datasets from both files and databases, ensuring a robust and scalable workflow.

Features:
Data Loading: Supports CSV, Excel, and database connections.

Automatic Data Processing:
Handles missing values in both target and non-target columns.
Automatically detects and encodes categorical variables.
Dynamically selects the target column or allows manual selection.

Model Training and Prediction:
Automatically detects regression or classification tasks based on the target variable.
Uses Random Forest as the default model.

Insights Generation:
Provides model performance metrics (e.g., R², Mean Squared Error).
Generates visualizations for feature importance and predictions.

Dashboard:
Creates an interactive dashboard using Dash for exploring insights.

Pipeline Scheduling:
Allows scheduling the pipeline to run at specific intervals (daily, weekly, monthly, or yearly).

Requirements:
Python 3.8+
Required Libraries:
pandas
numpy
sklearn
plotly
dash
sqlalchemy
schedule
openpyxl

Install dependencies using:
pip install -r requirements.txt

Usage
1. Clone the Repository
git clone https://github.com/AmanSagarRoy02/Aman-Sagar-Roy.git
cd DataScienceAutomation

3. Run the Pipeline
AUTOMATION.py

4. User Input Workflow
Step 1: Specify the data source type (file or database).
Step 2: Provide the file path or database connection details.
Step 3: The pipeline processes the data, trains a model, and generates insights.
Step 4: Visualize insights through the interactive dashboard.

5.Configuration
Target Column Selection
The pipeline attempts to auto-detect the target column based on:
High missing value ratio for the last columns.
Non-numeric data types for classification.
Numeric data types for regression.
Alternatively, specify a manual target column in the code.

6.Scheduling
The pipeline can be scheduled to run at specific intervals:
Daily
Weekly
Monthly
Yearly
Configure the scheduling during runtime when prompted.

7.Outputs:
Processed Dataset: Saved in output/processed_dataset.parquet.
Predictions: Saved in output/predictions.csv.
Insights: Saved in output/insights.json.

8.Dashboard
The dashboard provides:
Feature Correlation Heatmap
Feature Importance Bar Chart (for Random Forest models)
Prediction vs. Actual Scatter Plot
Histograms for Numeric Columns

To access the dashboard:
After running the pipeline, note the server URL in the logs (default: http://127.0.0.1:8050).
Open the URL in a web browser.

9.Logging
The pipeline logs every step to ensure traceability.
Logs are saved in logs/automation.log.

10.Troubleshooting
Target Column Error: If the target column is not correctly detected, specify it manually in the code (manual_target_col).
Missing Values: Ensure missing values in critical columns are handled correctly by the pipeline.
Dependencies: Verify that all required libraries are installed.

License
This project is licensed under the MIT License.(https://github.com/users/AmanSagarRoy02/projects/LICENSE)

Contact:
For support or inquiries:
Email:amansagarroy93@gmail.com
GitHub:https://github.com/AmanSagarRoy02
This README.md provides comprehensive information about the project and guides users on setup, usage, and troubleshooting.
