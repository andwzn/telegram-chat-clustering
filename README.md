# Telegram Chat Clustering

## Directory Structure

The project is organized into the following directories:

- `data`: Contains the datasets used for the analysis.
- `results`: Stores the outputs and results of the analysis.
- `notebook`: Includes Jupyter notebooks for different parts of the analysis.
- `util`: Holds additional scripts used in the project.


## Experiment Structure

The experiment itself is structured into two parts:

### 1. Data Exploration and Cleaning

- This part starts with an initial exploration of the data. Once the errors and irregularities  uncovered are fixed, a more in-depth exploratory analysis will be conducted.

- The code for this part can be found in the `01_data_cleaning_and_exploration` notebook.

### 2. Feature Engineering and Clustering

- TBD

- The code for this part will be provided in the `02_feature_engineering_and_clustering` notebook.

## Quickstart

1. **Clone the repository**: 
    ```bash
    git clone <repository-url>
    ```

2. **Navigate to the project directory**:
    ```bash
    cd <project-directory>
    ```

3. **Install the necessary dependencies for each notebook**:
    
    For 01_data_cleaning_and_exploration.ipynb:
    ```bash
    pip install -r requirements.txt
    ```

    For 02_feature_engineering_and_clustering.ipynb:
    ```bash
    pip install -r requirements_2.txt
    ```    

    Using Python 3.12.4 and seperate environments for each notebook is recommended.


4. **Add the data:**
    Add datasets (either as csv or sqlite-db) to their respective directories (see below)

5. **Run the notebooks**:
    Open the Jupyter notebooks in the `notebook` directory to explore the data and run the analysis.


## Adding the Data

### Adding the Datasets
- Due to privacy concerns, the datasets are not included in this repository. You can add the datasets in the following ways:
    1. **SQLite Databases**: If the data is provided as SQLite databases created by the Telegram data-collection and analysis suite TeleVision, place these databases in the `data/dbs` directory.
    2. **CSV Files**: If the data is already in CSV format, place these files in the `data/csv` directory.

### Converting the Dataset to CSV
- To convert all available SQLite databases in the `data/dbs` directory into CSV files, follow these steps:

    1. **Open a terminal or command prompt.**

    2. **Navigate to the root directory of the project.**
        ```bash
        cd <project-directory>
        ```

    3. **Run the following command:**
        ```sh
        python3 -m util.export_msg_from_db
        ```

- Please note that this process might take some time depending on the size of the databases. It also only works with dbs created using TeleVision.