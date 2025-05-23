import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.HuGaDB.corrupt_data_hugadb import corrupt_data, corrupt_labels  


def prepare_partitions(client_idx, trial_seed, sample_size, num_corrupted_clients,
                       train_files_pattern,noise_std, corruption_settings , label_encoder):

    train_file = train_files_pattern.format(client_idx)
    df_train_local = pd.read_csv(train_file).dropna(subset=['act'])
    
    df_train_local, _ = train_test_split(
        df_train_local,
        train_size=sample_size,
        random_state=trial_seed,
        stratify=df_train_local['act']
    )
    df_train_local = df_train_local.reset_index(drop=True).dropna()
    
    if client_idx <= num_corrupted_clients:
        X_train_local = df_train_local.drop('act', axis=1).values
        y_train_local = df_train_local['act'].values
        X_train_local = corrupt_data(
            X_train_local,
            corruption_prob=corruption_settings['corruption_prob'],
            nan_prob=corruption_settings['nan_prob'],
            noise_std=noise_std
        )
        y_train_local = corrupt_labels(
            y_train_local,
            corruption_prob=corruption_settings['label_corruption_prob']
        )

        df_train_local = pd.DataFrame(
            X_train_local,
            columns=pd.read_csv(train_file).drop('act', axis=1).columns
        )
        df_train_local['act'] = y_train_local
        df_train_local = df_train_local.dropna()
    

    X_train = df_train_local.drop('act', axis=1).values
    y_train = label_encoder.transform(df_train_local['act']) 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled, y_train