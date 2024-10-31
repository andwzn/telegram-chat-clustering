# Telegram Chat Clustering

This project explores novel approaches to clustering Telegram chats by evaluating various features derived from referenced made in Telegram messages. 

## Directory Structure

The project is organized into the following directories:

- `data`: Contains the datasets used for the analysis (e.g., auxiliary data, CSV files, preprocessed data, and models).

    - `auxiliary`: Contains auxiliary data
    - `csv`: Contains the raw and cleaned data as csv files.
    - `dbs`: Contains the data in a sqlite db (if provided).
    - `models`: Contains models used in the experiment.
    - `preprocessed`: Contains preprocessed data used for feature engineering and clustering.

- `results`: Stores outputs and results of the analysis.
    - `base_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated message embeddings.
    - `filtered_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated message embeddings, with original/forwarded pairs removed.

    - `log_combined_structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated message embeddings and structural embeddings normalized with the log function.
    - `onehot_combined_structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated message embeddings and structural embeddings encoded with one-hot encoding.
    - `combined_structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated message embeddings and structural embeddings that produced favorable results in pretrials.

    - `log_structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from structural embeddings normalized with the log function.
    - `onehot_structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from structural embeddings encoded with one-hot encoding.
    - `structural_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from structural embeddings that produced favorable results in pretrials.

    - `webpreview_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from aggregated web preview embeddings.
    - `combined_webpreview_embeddings`: Contains keywords, representative texts, visualizations, and the topic model created by applying BERTopic to chat representations generated from combined message and web preview embeddings.

    - `comparison`: Contains evaluation metrics and visualizations for all clusterings.

- `notebook`: Includes jupyter notebooks for different steps of the project.

- `util`: Holds scripts used in the project.

- `requirements`: Requirements of the conda envs for each notebook.


## Experiment Structure

The experiment itself is structured into three parts:

### 1. Data Exploration and Cleaning

- This part starts with an initial exploration of the data. Once the errors and irregularities  uncovered are fixed, a more in-depth exploratory analysis will be conducted.

- The code for this part can be found in the `01_data_cleaning_and_exploration` notebook.

### 2. Feature Engineering 

- This part focuses on creating various chat representations based on different types of references in the dataset.

- The code for this part will be provided in the `02_feature_engineering_and_clustering` notebook.

### 3. Clustering

- The final part includes clustering and evaluation of these features.

- The code for this part will be provided in the `03_clustering` notebook.


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

    Make sure you have Anaconda or Miniconda installed.

    Afterwards, you can run the following commands to create the environments for the different notebooks.
    
    For 01_data_cleaning_and_exploration.ipynb:
    ```bash
    conda create --name <env> --file <requirements_1.txt>
    ```

    For 02_feature_engineering.ipynb:
    ```bash
    conda create --name <env> --file <requirements_2.txt>
    ```

    For 03_clustering.ipynb:
    ```bash
    conda create --name <env> --file <requirements_3.txt>
    ```            

    If provided, you can also use the envs in the  `conda_envs`directory.

    Using Python 3.12.4 and seperate environments for each notebook is recommended. The text-files can be found in the `requirements` directory.


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

    3. **Run one of the following commands (depending on your setup):**
        ```sh
        python3 -m util.export_msg_from_db
        python -m util.export_msg_from_db
        ```

- Please note that this process might take some time depending on the size of the databases. It also only works with dbs created using TeleVision.