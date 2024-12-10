import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X_COLS = [
    'Manifest_ID',
    'TransportUnitSizeID',
    'TransportUnitTypeID',
    'Stop_Type_ID',
    'Stop_Sequence_in_Manifest',
    'Country_ID',
    'GPS_LATITUDE',
    'GPS_LONGITUDE'
]

PICKUP_STOP_TYPE_ID = 234
DELIVERY_STOP_TYPE_ID = 237

TARGET_CLASSES_MAX = 100 # rows with 101th 'Other' class will be dropped
COUNTRY_CLASSES_MAX = 10 # + 1 'Other' class


class CategoricalThresholder:
    def __init__(self, n_classes=10, threshold_class='Other'):
        self.n_classes = n_classes
        self.threshold_class = threshold_class
        self.valid_categories_ = None

    def fit(self, series):
        value_counts = series.value_counts()
        self.valid_categories_ = value_counts.nlargest(self.n_classes).index.tolist()
        return self

    def transform(self, series):
        if self.valid_categories_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        return series.apply(
            lambda x: x if x in self.valid_categories_ else self.threshold_class
        )

    def fit_transform(self, series):
        return self.fit(series).transform(series)


class DataController:
    def __init__(self):
        self.cat_thresholder_pickup = None
        self.cat_thresholder_delivery = None
        self.cat_thresholder_company_id = None
        self.pickup_dummies_columns = None
        self.delivery_dummies_columns = None
        self.gps_scaler = None
        self.raw_data_columns = None

    def save(self, filepath):
        """
        Serializes the DataController instance to a file.

        Parameters:
        - filepath: str, the path to the file where the instance will be saved.
        """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print(f"DataController saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Deserializes a DataController instance from a file.

        Parameters:
        - filepath: str, the path to the file from which the instance will be loaded.

        Returns:
        - DataController: the loaded instance.
        """
        with open(filepath, 'rb') as file:
            data_manager = pickle.load(file)
        print(f"DataController loaded from {filepath}")
        return data_manager

    def prepare_data(self, raw_dataframe, X_cols=X_COLS, is_training=True, force_include_target=None):
        """
        Prepares data for training or inference (and fits the data controller unless is_training is False).

        Parameters:
        - raw_dataframe: pd.DataFrame, the raw input data.
        - X_cols: list, the columns to select for processing.
        - is_training: bool, whether the function is called during training.
        - force_include_target: bool or None, forcefully include the target column. Used for scoring.

        Returns:
        - pd.DataFrame: the prepared dataset.
        """
        raw_dataframe = raw_dataframe.reset_index(drop=True)
        raw_dataframe = self._handle_missing_stop_sequence(raw_dataframe)
        df_X = raw_dataframe[X_cols].copy()

        if is_training:
            self.raw_data_columns = list(raw_dataframe.columns)

        # Process country data
        country_data = self._process_country_data(df_X, is_training)

        # Process GPS data
        gps_data = self._process_gps_data(df_X, is_training)

        # Process transport unit type and size
        transport_type_data = self._process_transport_type_data(df_X, is_training)
        transport_size_data = self._process_transport_size_data(df_X, is_training)

        # Combine all processed components
        dataset = pd.concat(
            [
                transport_type_data.drop('Manifest_ID', axis=1),
                transport_size_data.drop('Manifest_ID', axis=1),
                country_data.drop('Manifest_ID', axis=1),
                gps_data.drop('Manifest_ID', axis=1),
            ],
            axis=1
        )

        # For training, process COMPANY_ID
        if is_training:
            df_y = raw_dataframe.groupby('Manifest_ID').agg('first')['COMPANY_ID'].reset_index()
            self.cat_thresholder_company_id = CategoricalThresholder(n_classes=TARGET_CLASSES_MAX)
            df_y['COMPANY_ID'] = self.cat_thresholder_company_id.fit_transform(df_y['COMPANY_ID'])
            dataset = pd.concat([dataset, df_y['COMPANY_ID']], axis=1)
            dataset.drop(np.where(dataset['COMPANY_ID'] == 'Other')[0], inplace=True)
            dataset['COMPANY_ID'] = dataset['COMPANY_ID'].astype('int')
        # For scoring dataset
        elif force_include_target:
            df_y = raw_dataframe.groupby('Manifest_ID').agg('first')['COMPANY_ID'].reset_index()
            df_y['COMPANY_ID'] = self.cat_thresholder_company_id.transform(df_y['COMPANY_ID'])
            dataset = pd.concat([dataset, df_y['COMPANY_ID']], axis=1)
            dataset.drop(np.where(dataset['COMPANY_ID'] == 'Other')[0], inplace=True)
            dataset['COMPANY_ID'] = dataset['COMPANY_ID'].astype('int')

        dataset = dataset.reset_index(drop=True)

        return dataset

    def _handle_missing_stop_sequence(self, df):
        stop_seq_nan_idx = np.where(df['Stop_Sequence_in_Manifest'].isna())[0]
        for idx in stop_seq_nan_idx:
            df.loc[idx, 'Stop_Sequence_in_Manifest'] = 1 if df['Stop_Type_ID'][idx] == PICKUP_STOP_TYPE_ID else 2
        return df

    def _process_country_data(self, df_X, is_training):
        pickup_stops = df_X[df_X['Stop_Type_ID'] == PICKUP_STOP_TYPE_ID][['Manifest_ID', 'Country_ID']]
        delivery_stops = df_X[df_X['Stop_Type_ID'] == DELIVERY_STOP_TYPE_ID][['Manifest_ID', 'Country_ID']]

        # Threshold Country_ID for pickup and delivery stops
        if is_training:
            self.cat_thresholder_pickup = CategoricalThresholder(n_classes=COUNTRY_CLASSES_MAX)
            self.cat_thresholder_delivery = CategoricalThresholder(n_classes=COUNTRY_CLASSES_MAX)
            pickup_stops['Country_ID'] = self.cat_thresholder_pickup.fit_transform(pickup_stops['Country_ID'])
            delivery_stops['Country_ID'] = self.cat_thresholder_delivery.fit_transform(delivery_stops['Country_ID'])
        else:
            pickup_stops['Country_ID'] = self.cat_thresholder_pickup.transform(pickup_stops['Country_ID'])
            delivery_stops['Country_ID'] = self.cat_thresholder_delivery.transform(delivery_stops['Country_ID'])

        # Generate dummy columns
        pickup_dummies = pd.get_dummies(pickup_stops['Country_ID'], prefix='Country_Pickup')
        delivery_dummies = pd.get_dummies(delivery_stops['Country_ID'], prefix='Country_Delivery')

        if is_training:
            # Save column names during training
            self.pickup_dummies_columns = list(pickup_dummies.columns)
            self.delivery_dummies_columns = list(delivery_dummies.columns)
        else:
            # Ensure the same dummy columns are used for inference
            pickup_dummies = pickup_dummies.reindex(columns=self.pickup_dummies_columns, fill_value=False)
            delivery_dummies = delivery_dummies.reindex(columns=self.delivery_dummies_columns, fill_value=False)

        # Merge dummies back
        pickup_stops_dummy = pd.merge(
            left=pickup_stops,
            right=pickup_dummies,
            left_index=True,
            right_index=True
        )
        pickup_stops_dummy.drop('Country_ID', axis=1, inplace=True)

        delivery_stops_dummy = pd.merge(
            left=delivery_stops,
            right=delivery_dummies,
            left_index=True,
            right_index=True
        )
        delivery_stops_dummy.drop('Country_ID', axis=1, inplace=True)

        # Combine and aggregate country data
        country_data = pd.concat((
            pickup_stops_dummy,
            delivery_stops_dummy
        )).replace(np.nan, False).sort_index()

        country_data = country_data.groupby('Manifest_ID').agg('max').reset_index()

        # Add unique country count
        contries_unique_count = df_X.groupby('Manifest_ID').agg(unique_countries=("Country_ID", "nunique")).reset_index().drop('Manifest_ID', axis=1)

        country_data_full = pd.merge(
            left=country_data,
            right=contries_unique_count,
            left_index=True,
            right_index=True
        )

        return country_data_full

    def _process_gps_data(self, df_X, is_training):
        df_gps = df_X.sort_values(by=["Manifest_ID", "Stop_Sequence_in_Manifest"])
        gps_agg = df_gps.groupby("Manifest_ID").agg(
            start_lat=("GPS_LATITUDE", "first"),
            start_lon=("GPS_LONGITUDE", "first"),
            end_lat=("GPS_LATITUDE", "last"),
            end_lon=("GPS_LONGITUDE", "last"),
            min_lat=("GPS_LATITUDE", "min"),
            min_lon=("GPS_LONGITUDE", "min"),
            max_lat=("GPS_LATITUDE", "max"),
            max_lon=("GPS_LONGITUDE", "max"),
            mean_lat=("GPS_LATITUDE", "mean"),
            mean_lon=("GPS_LONGITUDE", "mean")
        ).reset_index()

        cols_to_scale = [
            "start_lat", "start_lon", "end_lat", "end_lon",
            "min_lat", "min_lon", "max_lat", "max_lon",
            "mean_lat", "mean_lon"
        ]
        if is_training:
            self.gps_scaler = MinMaxScaler()
            gps_agg[cols_to_scale] = self.gps_scaler.fit_transform(gps_agg[cols_to_scale])
        else:
            gps_agg[cols_to_scale] = self.gps_scaler.transform(gps_agg[cols_to_scale])

        return gps_agg

    def _process_transport_type_data(self, df_X, is_training):
        transport_type_dummies = pd.get_dummies(df_X['TransportUnitTypeID'], prefix='TransportUnitTypeID')

        if is_training:
            self.transport_type_dummy_columns = transport_type_dummies.columns
        else:
            transport_type_dummies = transport_type_dummies.reindex(columns=self.transport_type_dummy_columns, fill_value=False)

        df_transport = pd.merge(
            left=df_X[['Manifest_ID', 'TransportUnitTypeID']],
            right=transport_type_dummies,
            left_index=True,
            right_index=True
        )
        transport_agg = df_transport.groupby('Manifest_ID').agg('first').reset_index()
        return transport_agg.drop('TransportUnitTypeID', axis=1)


    def _process_transport_size_data(self, df_X, is_training):
        transport_size_dummies = pd.get_dummies(df_X['TransportUnitSizeID'], prefix='TransportUnitSizeID')

        if is_training:
            self.transport_size_dummy_columns = transport_size_dummies.columns
        else:
            transport_size_dummies = transport_size_dummies.reindex(columns=self.transport_size_dummy_columns, fill_value=False)

        df_transport = pd.merge(
            left=df_X[['Manifest_ID', 'TransportUnitSizeID']],
            right=transport_size_dummies,
            left_index=True,
            right_index=True
        )
        transport_agg = df_transport.groupby('Manifest_ID').agg('first').reset_index()
        return transport_agg.drop('TransportUnitSizeID', axis=1)
