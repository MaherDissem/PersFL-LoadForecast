import os
import glob
import pandas as pd
import tqdm

from config import config


def resample_load(
    df: pd.DataFrame, freq: str = "1H", method: str = "mean"
) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    if method == "mean":
        df = df.resample(freq).mean()
    else:
        raise NotImplementedError
    return df


def add_missing_timestamps_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    first_day = df.index[0].floor("D")
    last_day = df.index[-1].ceil("D")
    df = df.reindex(pd.date_range(first_day, last_day, freq="1H"))[:-1]
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # This function is called at runtime so that the min and max values are stored within the client object.
    min_val = df.min()
    max_val = df.max()
    df = (df - min_val) / (max_val - min_val)
    return df, min_val, max_val


def extract_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    pass


def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    pass


def preprocess_load(
    df_path: str, trg_feat: str = "Aggregate", freq: str = "1H"
) -> pd.DataFrame:

    # Load the dataset
    df = pd.read_csv(df_path, index_col=0)

    # Select the target feature
    if type(trg_feat) == str:
        trg_feat = [trg_feat]
    df = df[trg_feat]

    # Resample the data (taking The mean of the values within each hourly interval)
    df = resample_load(df, freq)

    # Add missing timestamps as NaN (raw data has interruptions periods)
    df = add_missing_timestamps_as_nan(df)

    # Check for missing values
    if df.isna().any().any():
        pass

    return df


if __name__ == "__main__":
    input_data_path = "data/raw/CLEAN_REFIT_081116"
    output_data_path = config.data_root

    os.makedirs(output_data_path, exist_ok=True)
    for csv_file in tqdm.tqdm(glob.glob(os.path.join(input_data_path, "*.csv"))):
        df = preprocess_load(csv_file)
        df.to_csv(
            os.path.join(output_data_path, os.path.basename(csv_file)),
            index_label="timestamp",
        )
