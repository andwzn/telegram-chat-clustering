# telegram-chat-clustering

# Quickstart 

## Add the datasets
Due to privacy concerns, the data is not provided in this repository. 
If the data is provided as sqlite-dbs created by he telegram data-collection and analysis suit TeleVision, it should be added to the `data/dbs` directory. If it's already in the csv-values, it can be added to `data/csv`

## Covert the dataset to csv
By running `python3 -m util.export_msg_from_db` from the root directory of the project, all available sqlite-dbs can be converted into csv-files. This only works for databases created using TeleVision and might take a while.