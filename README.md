# Telegram Chat Clustering

## Quickstart

### Adding the Datasets
Due to privacy concerns, the datasets are not included in this repository. You can add the datasets in the following ways:
1. **SQLite Databases**: If the data is provided as SQLite databases created by the Telegram data-collection and analysis suite TeleVision, place these databases in the `data/dbs` directory.
2. **CSV Files**: If the data is already in CSV format, place these files in the `data/csv` directory.

### Converting the Dataset to CSV
To convert all available SQLite databases in the `data/dbs` directory into CSV files, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the root directory of the project.
3. Run the following command:
    ```sh
    python3 -m util.export_msg_from_db
    ```

Please note that this process might take some time depending on the size of the databases. It also only works with dbs created using TeleVision.
