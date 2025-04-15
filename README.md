# Data Science Automation Pipeline

The **Data Science Automation Pipeline** is a comprehensive Python-based framework designed to streamline the processes involved in data ingestion, cleaning, feature engineering, model training, prediction, and advanced visualization. This tool is particularly useful for quickly prototyping machine learning workflows and generating insights from raw datasets using state-of-the-art libraries.

## Table of Contents

- Features
- Prerequisites
- Installation
- Usage
- Configuration
- Project Structure
- Troubleshooting
- Contributing
- License

## Features

- Multiple Data Sources  
  - Load data from local files (CSV, Excel, JSON, Parquet, TXT) or databases (SQLite, SQLAlchemy).
  
- Data Cleaning & Preprocessing  
  - Automatic handling of missing values, duplicate removal, JSON parsing, and datetime feature extraction.
  - Dropping of columns or rows based on user-defined thresholds.
  
- Target Column Selection  
  - Auto-detection of the target column using heuristic methods along with manual override.
  - Encoding for categorical target variables.

- Feature Engineering  
  - Automatic clustering of numeric features.
  - Seamless transformation of categorical and numerical features using a scikit-learn pipeline.

- Model Training & Prediction  
  - Supports both classification and regression tasks.
  - Implements RandomForest and KNN models.
  - Includes functionality for train/test splitting, stratified splits when possible, and performance evaluation.
  
- **Advanced Insights & Visualization**  
  - Generates confusion matrices, residual plots, feature importance charts, SHAP summary plots, partial dependence plots, and decision tree visualizations.
  - Writes model metrics and detailed reports to Excel.
  
- Logging & Error Handling  
  - Robust logging using Python’s `logging` module with detailed error tracing for debugging.
  
- Dashboard and Scheduling  
  - Integration with Dash for creating interactive dashboards.
  - Utilities for scheduling tasks and graceful shutdown on receiving system signals.

##Prerequisites

Before running the pipeline, ensure you have the following installed:

- Python 3.7+
- Essential Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `sqlalchemy`
  - `dash`
  - `plotly`
  - `shap` (optional for SHAP analysis)
  - Additional packages: `schedule`, `logging`

You can install the necessary libraries with:

bash
pip install numpy pandas matplotlib seaborn scikit-learn sqlalchemy dash plotly shap schedule


> **Note: If you do not plan to use SHAP analysis, the pipeline will continue functioning but will log a warning.

## Installation

1. Clone the Repository:

   bash
   git clone https://github.com/AmanSagarRoy02
   cd ds-automation-pipeline
   

2. Create a Virtual Environment (Recommended):

   bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   

3. Install Dependencies:

   bash
   pip install -r requirements.txt
   

   If a `requirements.txt` file is not provided, use the installation command above under prerequisites.

## Usage

1. Run the Pipeline:

   Execute the main automation script:

   ```bash
   python AUTOMATION.py
   ```

2. Data Source Selection:

   - **Local File:** If you choose option 1, the script will prompt for the file path (supporting CSV, Excel, JSON, Parquet, and TXT formats).
   - **Database:** If you choose option 2, you can connect to a SQLite or SQLAlchemy database and input an SQL query to retrieve the dataset.

3. Interactive Prompts:

   - The tool may ask for the target column if not auto-detected.
   - You can accept the suggested target or provide an alternative column name.
   - During data loading, cleaning, and processing, the tool logs detailed messages to `pipeline.log`.

4. **Model Training and Prediction:

   - The pipeline builds preprocessing pipelines, applies train/test splits, trains both RandomForest and KNN models, and evaluates performance.
   - It handles both classification and regression tasks based on the target column's data type.

5. Insights & Visualizations:

   - Visual outputs (e.g., confusion matrix, residual plots, feature importance) are saved in an `output` folder within the project directory.
   - Metrics are saved as an Excel file (`metrics.xlsx`), and predictions are written to a CSV file (`predictions.csv`).

6. Dashboard Integration:

   - If dashboard features are enabled, launch the interactive dashboard provided by Dash for additional data exploration.

## Configuration

The script can be customized by editing parameters within the source code:

- **Logging:** Adjust log level (INFO, DEBUG) in the `logging.basicConfig` section.
- **Data Cleaning:** Tweak thresholds for dropping columns/rows, or customize JSON parsing and datetime extraction.
- **Model Parameters:** Modify hyperparameters for RandomForest and KNN models in the pipeline (e.g., `n_neighbors`, `n_jobs`).
- **Output Directory:** Change the location where processed data, models, and visualizations are saved by editing the `visualization_dir` variable.

## Project Structure

```
ds-automation-pipeline/
├── AUTOMATION.py         # Main automation script
├── requirements.txt      # List of dependencies (if provided)
├── pipeline.log          # Log file generated during execution
├── output/               # Folder where processed data, plots, and metrics are saved
└── README.md             # This README file
```

## Troubleshooting

- **File Not Found Errors:**  
  Double-check the file path provided during data source selection.

- **Missing Library Warnings:**  
  If SHAP or any other library is not found, install the missing package via pip.
  
- **Data Processing Issues:**  
  Check `pipeline.log` for detailed error messages. Ensure your data conforms to expected formats.

- **Performance Issues:**  
  For large datasets, consider reducing data size or adjusting model complexity. Detailed logging can help pinpoint bottlenecks.

License
This project is licensed under the MIT License.(https://github.com/users/AmanSagarRoy02/projects/LICENSE)

Contact:
For support or inquiries:
Email:amansagarroy93@gmail.com
GitHub:https://github.com/AmanSagarRoy02
This README.md provides comprehensive information about the project and guides users on setup, usage, and troubleshooting.
