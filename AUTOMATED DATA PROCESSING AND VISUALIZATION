import pandas as pd
import matplotlib.pyplot as plt
import schedule
import time
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('attrition_analysis.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

def load_and_clean_data(file_path):
    logger.info("Loading and cleaning data...")
    try:
        excel_data = pd.ExcelFile(file_path)
        df = excel_data.parse('Berger Paints')
        df_cleaned = df.iloc[1:].reset_index(drop=True)

        date_columns = ['Date_of_Join', 'Date_of_Leaving']
        for col in date_columns:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], format='%d/%m/%Y', errors='coerce')

        df_cleaned['Attrition_Year_Month'] = pd.to_datetime(
            df_cleaned['Client_Attrition_Month'] + ' ' + df_cleaned['Date_of_Leaving'].dt.year.astype(str),
            format='%B %Y', errors='coerce')

        random.seed(42)
        np.random.seed(42)

        reasons_secondary = ['Career Growth', 'Salary', 'Personal Reasons', 'Relocation', 'Work-Life Balance',
                             'Health Issues', 'Better Opportunity', 'Company Culture']
        reasons_tertiary = ['Work Environment', 'Supervisor Issues', 'Company Policies', 'Commute Distance', 'Workload',
                            'Team Dynamics']

        for col in ['Secondary_Reason_for_Leaving_Node_1', 'Tertiary_Reason_for_Leaving_Node_1']:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

        missing_attrition_dates = df_cleaned['Attrition_Year_Month'].isna().sum()
        df_cleaned.loc[df_cleaned['Attrition_Year_Month'].isna(), 'Attrition_Year_Month'] = pd.to_datetime(
            np.random.choice(pd.date_range("2023-04-01", "2023-12-31"), missing_attrition_dates)
        )

        logger.info("Saving cleaned data back to the Excel file...")
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_cleaned.to_excel(writer, index=False, sheet_name='Berger Paints')

        logger.info("Data cleaning complete.")
        return df_cleaned
    except Exception as e:
        logger.error("Error in load_and_clean_data: {}".format(e))
        raise

def aggregate_data(df_cleaned):
    logger.info("Aggregating data...")
    attrition_summary = df_cleaned.groupby('Attrition_Year_Month').size().reset_index(name='Number_of_Employees_Left')

    reasons_summary = df_cleaned[['Secondary_Reason_for_Leaving_Node_1', 'Tertiary_Reason_for_Leaving_Node_1']].apply(
        pd.Series.value_counts).fillna(0).astype(int)

    logger.info("Data aggregation complete.")
    return attrition_summary, reasons_summary

def visualize_data(attrition_summary, reasons_summary):
    logger.info("Visualizing data...")

    if not attrition_summary.empty:
        attrition_summary_sorted = attrition_summary.sort_values('Attrition_Year_Month')
        plt.figure(figsize=(10, 6))
        plt.plot(attrition_summary_sorted['Attrition_Year_Month'],
                 attrition_summary_sorted['Number_of_Employees_Left'], marker='o')
        plt.title('Number of Employees Who Left Each Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Employees Left')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('attrition_plot.png')
        plt.show()
    else:
        logger.warning("Attrition summary is empty. No data to plot.")

    if not reasons_summary.empty:

        reasons_summary_corrected = reasons_summary.loc[reasons_summary.index != 'Unknown']

        reasons_summary_top = reasons_summary_corrected.sum(axis=1).nlargest(10)
        plt.figure(figsize=(10, 6))
        reasons_summary_top.plot(kind='barh', color='skyblue')
        plt.title('Top 10 Reasons for Leaving')
        plt.xlabel('Number of Employees')
        plt.ylabel('Reason')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reasons_plot.png')
        plt.show()
    else:
        logger.warning("Reasons summary is empty. No data to plot.")

    logger.info("Visualization complete.")

def main():
    logger.info("Starting main process...")
    file_path = r"C:/Users/amans/OneDrive/Documents/Berger Paint Attrition Analysis April_2023 To December_2023.xlsx"
    try:
        df_cleaned = load_and_clean_data(file_path)
        attrition_summary, reasons_summary = aggregate_data(df_cleaned)
        visualize_data(attrition_summary, reasons_summary)
        logger.info("Data processing and visualization complete.")
    except Exception as e:
        logger.error("Error in main function: {}".format(e))

def run_daily():
    logger.info("Running scheduled task...")
    main()

schedule.every().day.at("00:00:00").do(run_daily)

print("Scheduler setup complete. Waiting for the scheduled time...")

while True:
    schedule.run_pending()
    time.sleep(30)
