import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

class Main(object):
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.feature_columns = None
        self.artifacts_directory = os.environ.get('artifacts_directory', os.path.join(self.cur_dir, 'artifacts'))
        self.keep_training = os.environ.get('keep_training', 'False')
        self.model = None
        print("Main object initialized.")
                
        if os.path.isfile(os.path.join(self.cur_dir, './model/linear_regression_model.pkl')):
            self.model = joblib.load(os.path.join(self.cur_dir, './model/linear_regression_model.pkl'))
    
    def train(self, training_directory):
        print(training_directory)
        if os.path.isfile(os.path.join(self.cur_dir, training_directory, 'your.csv')):
            data = pd.read_csv(os.path.join(self.cur_dir, training_directory, 'your.csv'), header=0, encoding="utf-8")
            self.feature_columns = data.drop(data.columns[-1], axis=1).columns
            X, y = data[self.feature_columns].values, data[data.columns[-1]].values
        else:
            X, y = self.load_data(training_directory)

        if self.model is None or self.keep_training == 'True':
            self.model = self.build_model(X, y)
        else:
            self.model.fit(X, y)
            print("Model trained.")


    def evaluate(self, evaluation_directory):
        if os.path.isfile(os.path.join(self.cur_dir, evaluation_directory, 'evaluate.csv')):
            data = pd.read_csv(os.path.join(self.cur_dir, evaluation_directory, 'evaluate.csv'), header=0, encoding="utf-8")
            self.feature_columns = data.drop(data.columns[-1], axis=1).columns
            X, y = data[self.feature_columns].values, data[data.columns[-1]].values
        else:
            X, y = self.load_data(evaluation_directory)
        return self.model.score(X, y)

    def save(self):
        joblib.dump(self.model, os.path.join(self.cur_dir, './model/linear_regression_model.pkl'))

    def load_data(self, data_directory):
        df_list = []
        for filename in os.listdir(os.path.join(self.cur_dir, data_directory)):
            if not os.path.isfile(os.path.join(self.cur_dir, data_directory, filename)):
                continue
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(self.cur_dir, data_directory, filename), header=0, encoding="utf-8")
                    df_list.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty CSV file found and ignored: {filename}")
    
        if not df_list:
            raise ValueError("No CSV files found in the specified directory.")

        data = pd.concat(df_list, axis=0)
        self.feature_columns = data.drop(data.columns[-1], axis=1).columns

        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns
        non_numeric_cols = [col for col in data.columns if col not in numeric_cols]

        # Handle non-numeric columns (e.g., date columns)
        for col in non_numeric_cols:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                # Convert date columns to numeric representation (e.g., timestamp)
                data[col] = data[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
            else:
                # Handle categorical columns (e.g., encoding)
                data[col] = data[col].astype('category').cat.codes

        X = data[self.feature_columns].values
        y = data[data.columns[-1]].values
        
        return X, y

    def process_data(self, data_directory):
        X, y = self.load_data(data_directory)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
        train_df = pd.DataFrame(X_train, columns=self.feature_columns)
        train_df[y_train] = y_train
        train_df.to_csv(os.path.join(self.cur_dir, data_directory, 'training', 'your.csv'), index=False)

        test_df = pd.DataFrame(X_test, columns=self.feature_columns)
        test_df[y_test] = y_test
        test_df.to_csv(os.path.join(self.cur_dir, data_directory, 'test', 'evaluate.csv'), index=False)



if __name__ == "__main__":
    main_obj = Main()
    main_obj.train()
    accuracy = main_obj.evaluate()
    print("Accuracy:", accuracy)
    main_obj.save()
    print("Process completed.")