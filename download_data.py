from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path('data')


if __name__ == '__main__':
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Load the data
    print('Loading the data...', end='', flush=True)
    file_path = DATA_PATH / "bera.csv"
    X_df = pd.read_csv(file_path)

    # Clean the data
    X_df = X_df.drop_duplicates()
    X_df = X_df.drop(columns=['commentaire', 'url_telechargement'])

    # Preprocess the data
    X_df = X_df.dropna(subset=['risque1'])
    X_df = X_df[X_df['risque1'] != -1]
    X_df['risque2'] = X_df['risque2'].fillna(X_df['risque1'])

    # Create the target (ou bien max entre risque1 et risque2 ?)
    y = X_df['risque1']

    # Split the data
    X_train, X_test = train_test_split(
        X_df, test_size=0.2, random_state=57, shuffle=True,
        stratify=y
    )

    # Save the data
    X_train.to_csv(DATA_PATH / 'X_train.csv', index=False)
    X_test.to_csv(DATA_PATH / 'X_test.csv', index=False)
    print('done')
