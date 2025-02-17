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
    X_df = X_df.drop(columns=['loc1', 'evolurisque2', 'loc2', 'commentaire',
                              'url_telechargement'])
    # columns = ['date', 'massif', 'risque1', 'risque2','altitude', 'evolurisque1',
    #            'risque_maxi', '00_temps', '00_mer_de_nuages',
    #            '00_limite_pluie_neige', '00_isotherme_0', '00_isotherme_moins_10',
    #            '00_altitude_vent_1', '00_altitude_vent_2',
    #            '00_direction_vent_altitude_1', '00_vitesse_vent_altitude_1',
    #            '00_direction_vent_altitude_2', '00_vitesse_vent_altitude_2',
    #            '06_temps', '06_mer_de_nuages', '06_limite_pluie_neige',
    #            '06_isotherme_0', '06_isotherme_moins_10', '06_altitude_vent_1',
    #            '06_altitude_vent_2', '06_direction_vent_altitude_1',
    #            '06_vitesse_vent_altitude_1', '06_direction_vent_altitude_2',
    #            '06_vitesse_vent_altitude_2', '12_temps', '12_mer_de_nuages',
    #            '12_limite_pluie_neige', '12_isotherme_0', '12_isotherme_moins_10',
    #            '12_altitude_vent_1', '12_altitude_vent_2',
    #            '12_direction_vent_altitude_1', '12_vitesse_vent_altitude_1',
    #            '12_direction_vent_altitude_2', '12_vitesse_vent_altitude_2',
    #            'precipitation_neige_veille_altitude', 'precipitation_neige_veille_epaisseur'
    #         ]

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
