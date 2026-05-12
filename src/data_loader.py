import os
from pyclbr import Class
import pandas as pd

class AnimeDataLoader:
    def __init__(self, original_csv:str, processed_csv:str):
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self):
        if not os.path.exists(self.original_csv):
            raise FileNotFoundError(f"Original data file not found at {self.original_csv}")

        try:
            data = pd.read_csv(self.original_csv, encoding='utf-8', on_bad_lines='skip').dropna()

            required_columns = {'Name', 'Genres', 'sypnopsis' }

            if not required_columns.issubset(data.columns):
                missing_cols = required_columns - set(data.columns)
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            data['combined_cols'] = data['Name'] + ' ' + data['Genres'] + ' ' + data['sypnopsis']
            
            data[['combined_cols']].to_csv(self.processed_csv, index=False, encoding='utf-8')

            return self.processed_csv
        
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
