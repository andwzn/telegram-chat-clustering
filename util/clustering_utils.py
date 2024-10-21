import os
import datetime
import re
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from typing import Optional, Tuple, Dict, Union, List
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer


def load_emoji_list(file_paths: list[str]) -> list[str]:
        """
        Load a list of all emoji from the given file paths.
        Args:
            file_paths (list): A list of file paths to load emoji sequences from.
        Returns:
            list: A list of unicode sequences representing the loaded emoji sequences.
        """
        
        unicode_list = []

        # match lines with unicode, including ranges like 231A..231B 
        range_pattern = re.compile(r"([0-9A-Fa-f]{4,6})\.\.([0-9A-Fa-f]{4,6})\s*;\s*")
        code_point_pattern = re.compile(r"([0-9A-Fa-f]{4,6}(?:\s[0-9A-Fa-f]{4,6})*)\s*;\s*")

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            for line in lines:
                range_match = range_pattern.match(line)
                
                # add elements of ranges as individual codes to list
                if range_match:
                    start_code, end_code = range_match.groups()
                    start_int = int(start_code, 16)
                    end_int = int(end_code, 16)
                    unicode_list.extend([chr(code) for code in range(start_int, end_int + 1)])
                else:
                    code_match = code_point_pattern.match(line)
                    if code_match:
                        code_points = code_match.group(1)       
                        code_point_list = code_points.split()
                        # create zwj sequences by combining all code points
                        unicode_list.append(''.join([chr(int(code, 16)) for code in code_point_list]))
        print("Emoji sequences loaded")
        return unicode_list
    

def print_log(module, step, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{timestamp} - {module} - {step} - {message}")
    
    
def apply_bertTopic(chat_embeddings: pd.Series, chat_texts: pd.Series, used_embedding_model: SentenceTransformer, hdbscan_model=None, verbose=True): 

    # prepare the embeddings for dimensionality reduction by stacking them
    chat_embeddings = np.vstack(chat_embeddings)

    if verbose:
        print_log("apply_bertTopic", "Preparation", "Completed ✓")

    # create your representation model
    representation_model = KeyBERTInspired() #TODO: Configure?

    #TODO: Random state?

    # initiate the BERTopic model
    docs = chat_texts.tolist()
        
    #cluster_model = KMeans(n_clusters=14) #9 #15->gut

    if hdbscan_model is None:
        topic_model = BERTopic(embedding_model=used_embedding_model, 
                            verbose=verbose, 
                            calculate_probabilities=True, 
                            representation_model=representation_model,
                            )#hdbscan_model=cluster_model 
    else:
        topic_model = BERTopic(embedding_model=used_embedding_model, 
                            verbose=verbose, 
                            calculate_probabilities=True, 
                            representation_model=representation_model,
                            hdbscan_model=hdbscan_model)

    if verbose:
        print_log("apply_bertTopic", "Loading Model", "Completed ✓")
        
    # fit the model to the reduced embeddings
    topics, propabilities = topic_model.fit_transform(embeddings = chat_embeddings, documents = docs)
    if verbose:
        print_log("apply_bertTopic", "Model Fitting", "Completed ✓")

    return topics, propabilities, topic_model



def get_evaluations(chat_embeddings: pd.Series, propabilities: np.ndarray, topic_model: BERTopic, text_aggregations: pd.Series) -> Union[float, float, float, int, int]:
    """
    Get several evaluation metrics for a given topic model.
    Parameters:
        chat_embeddings (pd.Series): The chat embeddings.
        propabilities (pd.Series): The propabilities.
        topic_model (BERTopic): The topic model.
        text_aggregations (pd.Series): The text aggregations.
    Returns:
        tuple: A tuple containing the coherence score, silhouette score, davies bouldin score, topic count and noise count.
    """
    
    # get the document info and topics
    document_info = topic_model.get_document_info(text_aggregations)
    topics = document_info['Topic']
    
    # prepare the embeddings by stacking them
    chat_embeddings = np.vstack(chat_embeddings)    

    # get the silhouette score while ignoring the "Other" topic
    valid_indices_filter = topics != -1
    filtered_embeddings = chat_embeddings[valid_indices_filter]
    filtered_topics = topics[valid_indices_filter]
    silhouette_score_result = silhouette_score(X=filtered_embeddings, labels=filtered_topics)
    #print(f'Silhouette Score: {silhouette_score_result}')
    
    # get davies bouldin score
    davies_bouldin_score_result = davies_bouldin_score(X=filtered_embeddings, labels=filtered_topics)
    #print(f'Davies-Bouldin Score: {davies_bouldin_score_result}')
    
    # calculate the number of topics found
    topic_count = len(np.unique(topics))
    #print(f'Topic Count: {topic_count}')
    
    # calculate the number of noise points
    noise_count = len(topics[topics == -1])
    #print(f'Noise Count: {noise_count}')
    
    # calculate the coherence score
    # preprocess documents is not necessary, as we already preprocessed the text aggregations
    
    # get the vectorizer
    vectorizer = topic_model.vectorizer_model
    tokenizer = vectorizer.build_tokenizer()
    
    # extract features for coherence evaluation
    #words = vectorizer.get_feature_names()
    tokens = [tokenizer(doc) for doc in text_aggregations] 
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                for topic in range(len(set(topics))-1)]

    # calculate the coherence score
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v') 
    coherence_score_result = coherence_model.get_coherence()    
    
    #print(f'Coherence Score: {coherence_score_result}')

    return coherence_score_result, silhouette_score_result, davies_bouldin_score_result, topic_count, noise_count



def run_experiment(
    chat_embeddings: pd.Series, 
    chat_texts: pd.Series, 
    n: int, 
    topic_model_dir_path: str,
    feature_name: str,
    used_embedding_model: SentenceTransformer,
    hdbscan_model: Optional[object] = None
) -> Tuple[Dict[str, float], List[int], List[float], BERTopic]:
    
    """
    Run the BERTopic model multiple times, calculate average evaluation metrics and return and save the "most average" model for manual inspection.
    
    Parameters:
    - chat_embeddings (pd.Series): The chat embeddings.
    - chat_texts (pd.Series): The aggregated chat texts.
    - n (int): Number of times to run the model.
    - topic_model_dir_path (str): Directory path to save topic models.
    - hdbscan_model (optional): A pre-defined HDBSCAN model to use. If None, a default model will be created.

    Returns:
    - avg_evaluation_metrics (dict): A dictionary containing average evaluation metrics.
    - selected_topics (List[int]): The topics assigned by the most average model.
    - selected_probabilities (List[float]): The probabilities for each document's topic assignment by the most average model.
    - avg_model (BERTopic): The BERTopic model that is closest to the average metrics.
    """
    
    coherence_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []
    topic_counts = []
    noise_counts = []
    
    # save topics and propabilities of each model to return them for the most average model later on
    topics_list = []
    propabilities_list = []

    for i in range(n):
        
        print(f"\n#### Running Model {i+1}/{n}... ####")
        
        print_log("run_experiment", "Fitting Model", "Fitting Model")
        topics, propabilities, topic_model = apply_bertTopic(chat_embeddings, chat_texts, used_embedding_model, hdbscan_model) #Remove hdbscan_model? #TODO: propabilities
        topics_list.append(topics)
        propabilities_list.append(propabilities)
        print_log("run_experiment", "Fitting Model", "Completed ✓")

        print_log("run_experiment", "Evaluating Model", "Calculating Evaluation Metrics")
        # calculate evaluation metrics
        (cs_base_embeddings,
        ss_base_embeddings,
        db_base_embeddings,
        topic_count_base_embeddings, 
        noise_base_embeddings) = get_evaluations(chat_embeddings, 
                                                propabilities, 
                                                topic_model, 
                                                chat_texts)
        
        # append the evaluation metrics
        coherence_scores.append(cs_base_embeddings)
        silhouette_scores.append(ss_base_embeddings)
        davies_bouldin_scores.append(db_base_embeddings)
        topic_counts.append(topic_count_base_embeddings)
        noise_counts.append(noise_base_embeddings)
        print_log("run_experiment", "Evaluating Model", "Completed ✓")

        print_log("run_experiment", "Saving Topic Model", "Saving Model")
        # save model
        topic_model_path =  os.path.join(topic_model_dir_path, f"{feature_name}_topic_model_{i}")
        os.makedirs(topic_model_dir_path, exist_ok=True)
        topic_model.save(topic_model_path)
        print_log("run_experiment", "Saving Topic Model", "Completed ✓")

    print(f"\n#### Calculating Averages ####")
    print_log("run_experiment", "Calculate Average Evaluation Metrics", "Calculating")    
    # calculate average evaluation metrics
    avg_coherence_scores = np.mean(coherence_scores)
    avg_silhouette_scores = np.mean(silhouette_scores)
    avg_davies_bouldin_scores = np.mean(davies_bouldin_scores)
    avg_topic_counts = np.mean(topic_counts)
    avg_noise_counts = np.mean(noise_counts)
    
    # create dictionary for the evaluation metrics
    avg_evaluation_metrics = {
        "avg_coherence_scores": avg_coherence_scores,
        "avg_silhouette_scores": avg_silhouette_scores,
        "avg_davies_bouldin_scores": avg_davies_bouldin_scores,
        "avg_topic_counts": avg_topic_counts,
        "avg_noise_counts": avg_noise_counts
    }
    print_log("run_experiment", "Calculate Average Evaluation Metrics", "Completed ✓")
    
    print_log("run_experiment", "Found Most Average Model", "Calculating")
    
    # find the model with evaluation results closests to the average evaluation metrics
    # convert evaluation metrics to a matrix for easier comparison with the average value vector
    evaluation_matrix = np.stack([coherence_scores, silhouette_scores, davies_bouldin_scores, topic_counts, noise_counts])
    average_value_vector = np.array([avg_coherence_scores, avg_silhouette_scores, avg_davies_bouldin_scores, avg_topic_counts, avg_noise_counts])
    average_value_vector = average_value_vector[:, np.newaxis]

    # find the model with the smallest difference to the average evaluation metrics
    differences = np.abs(evaluation_matrix - average_value_vector)
    smallest_diff_idx = np.argmin(np.sum(differences, axis=0))
    
    # load the model with the smallest difference to the average evaluation metrics
    average_model_path =  os.path.join(topic_model_dir_path, f"{feature_name}_topic_model_{smallest_diff_idx}")
    avg_model = BERTopic.load(average_model_path)
    print_log("run_experiment", "Found Most Average Model", "Completed ✓")
    
    # delete all other models
    print_log("run_experiment", "Deleted Remaining Models", "Deleting")
    for i in range(n):
        if i != smallest_diff_idx:
            os.remove(os.path.join(topic_model_dir_path, f"{feature_name}_topic_model_{i}"))
            
    # rename the model 
    os.rename(average_model_path, os.path.join(topic_model_dir_path, f"avg_{feature_name}_topic_model"))
    print_log("run_experiment", "Deleted Remaining Models", "Completed ✓")
        
    return avg_evaluation_metrics, topics[smallest_diff_idx], propabilities[smallest_diff_idx], avg_model


# def get_representative_texts(df: pd.DataFrame, 
#                              topic_model: BERTopic, 
#                              topic_vectors: pd.Series, 
#                              chat_vectors: pd.Series,
#                              n: int,
#                              feature_name: str,
#                              text_column: str,
#                              text_embeddings_column: str,
#                              text_preprocessed_column: str) -> Dict[int, List[str]]:
#     """Retrieve representative text for each topic from the given DataFrame.

#     Parameters:
#         df (pd.DataFrame): the DataFrame containing message data, including chat IDs and message vectors.
#         topic_model (BERTopic): the topic model 
#         topic_vectors (pd.Series): a pd.Series where the index represents topic IDs and values are their corresponding mean vectors.
#         chat_vectors (pd.Series): the chat representations used in the topic model.
#         n (int): the number of top representative messages to retrieve for each topic.
#         feature_name (str): the name of the feature(s) used to create the chat representations
#         text_column (str): the name of the column containing the text data
#         text_embeddings_column (str): the name of the column containing the text embeddings
#         text_preprocessed_column (str): the name of the column containing the preprocessed text data

#     Returns:
#         dict: A dictionary where keys are topic IDs and values are Series of the top representative messages for each topic.
#     """    
    
#     print_log("get_representative_texts", "Preperation", "Creating Chat/Topic Map")
    
#     # create a map with chat ids and their corresponding topics
#     topics = topic_model.topics_
#     chat_ids= chat_vectors.index
#     id_topic_map = pd.Series(topics, index=chat_ids, name='Topic')

#     print_log("get_representative_texts", "Preperation", "Add Topic Assignments to DataFrame")
#     # add the topic assignment of the base model to the message in the dataframe
#     df[f"{feature_name}_topic_assignment"] = df["telegram_chat_id"].map(id_topic_map)

#     # check if the number of unique combinations is equal to the number of chat ids (each chat id should have exactly one topic assignment)
#     unique_combinations = df[["telegram_chat_id", f"{feature_name}_topic_assignment"]].drop_duplicates()
#     assert unique_combinations.shape[0] == df["telegram_chat_id"].nunique()
    
#     print_log("get_representative_texts", "Preperation", "Completed ✓")
    
#     top_topic_messages = {}

#     # iterate over all topics (excluding the "Other" topic)
#     topics = topic_model.get_topic_info()["Topic"]
#     topics = topics.values[topics.values != -1]
#     for topic in topics:
#         print_log("get_representative_texts", "Find Representative Texts", f"Getting Texts for Topic {topic}")
        
#         # get all messages sent in groups assigned to a specific topic (excluding empty messages)
#         topic_messages_filter = (df[f"{feature_name}_topic_assignment"] == topic) & ((df[text_column] != "") | pd.isna(df[text_column]))
#         topic_msgs = df[topic_messages_filter]

#         # get the topic vector of the topic
#         topic_vector = topic_vectors[topic]

#         # check if the message vectors and topic vectors have the same dimension
#         assert topic_msgs.iloc[0]["message_vector"].shape == topic_vector.shape
        
#         def calculate_similarity(row: pd.Series, topic_vector: pd.Series, text_embeddings_column: str) -> float:
#             """
#             Calculate the cosine similarity between a text vector and a topic vector
#             Parameters:
#                 row (pd.Series): the row containing the text vector
#                 topic_vector (pd.Series): the topic vector
#                 text_embeddings_column (str): the name of the column containing the text embeddings
#             Returns:
#                 float: the cosine similarity between the text vector and the topic vector
#             """
#             message_vector = np.array(row[text_embeddings_column]).reshape(1, -1)  
#             return cosine_similarity(message_vector, topic_vector)[0][0]

#         # reshape the topic vector to fit the specifications of the cosine_similarity function
#         topic_vector = topic_vector.reshape(1, -1)

#         # calculate the cosine similarity between the message vectors and the topic vector
#         topic_msgs["similarity"] = topic_msgs.apply(lambda x: calculate_similarity(x, topic_vector, text_embeddings_column), axis=1)

#         # drop duplicates to avoid displaying the same message multiple times. We use the preprocessing column to identify duplicates, as the original message text might contain insignificant differences like whitespaces
#         topic_msgs = topic_msgs.drop_duplicates(subset=[text_preprocessed_column])

#         # sort the messages by similarity and display the most similar messages
#         topic_msgs = topic_msgs.sort_values(by="similarity", ascending=False)

#         top_messages = topic_msgs[text_column].head(n)   
        
#         top_topic_messages[topic] = list(top_messages.values)
        
#         print_log("get_representative_texts", "Find Representative Texts", f"Completed ✓")
        
#     return top_topic_messages

def get_representative_texts(df: pd.DataFrame, 
                             topic_model: BERTopic, 
                             topic_vectors: pd.Series, 
                             chat_vectors: pd.Series,
                             n: int,
                             feature_name: str,
                             text_column: str,
                             text_embeddings_column: str,
                             text_preprocessed_column: str,
                             add_structural_info: bool = False,
                             structural_embedding_chat_map: pd.Series = None) -> Dict[int, List[str]]:
    """Retrieve representative text for each topic from the given DataFrame.

    Parameters:
        df (pd.DataFrame): the DataFrame containing message data, including chat IDs and message vectors.
        topic_model (BERTopic): the topic model 
        topic_vectors (pd.Series): a pd.Series where the index represents topic IDs and values are their corresponding mean vectors.
        chat_vectors (pd.Series): the chat representations used in the topic model.
        n (int): the number of top representative messages to retrieve for each topic.
        feature_name (str): the name of the feature(s) used to create the chat representations
        text_column (str): the name of the column containing the text data
        text_embeddings_column (str): the name of the column containing the text embeddings
        text_preprocessed_column (str): the name of the column containing the preprocessed text data
        add_structural_info (bool): a boolean indicating whether structural vectors of a chat should be added to the message vectors
        structural_embedding_chat_map(pd.Series): a pd.Series where the index represents chat IDs and values are the structural embeddings of the chat

    Returns:
        dict: A dictionary where keys are topic IDs and values are Series of the top representative messages for each topic.
    """    
    
    print_log("get_representative_texts", "Preperation", "Creating Chat/Topic Map")
    
    # create a map with chat ids and their corresponding topics
    topics = topic_model.topics_
    chat_ids= chat_vectors.index
    id_topic_map = pd.Series(topics, index=chat_ids, name='Topic')

    print_log("get_representative_texts", "Preperation", "Add Topic Assignments to DataFrame")
    # add the topic assignment of the base model to the message in the dataframe
    df[f"{feature_name}_topic_assignment"] = df["telegram_chat_id"].map(id_topic_map)

    # check if the number of unique combinations is equal to the number of chat ids (each chat id should have exactly one topic assignment)
    unique_combinations = df[["telegram_chat_id", f"{feature_name}_topic_assignment"]].drop_duplicates()
    assert unique_combinations.shape[0] == df["telegram_chat_id"].nunique()
    
    print_log("get_representative_texts", "Preperation", "Completed ✓")
    
    top_topic_messages = {}

    # iterate over all topics (excluding the "Other" topic)
    topics = topic_model.get_topic_info()["Topic"]
    topics = topics.values[topics.values != -1]
    for topic in topics:
        print_log("get_representative_texts", "Find Representative Texts", f"Getting Texts for Topic {topic}")
        
        # get all messages sent in groups assigned to a specific topic (excluding empty messages)
        topic_messages_filter = (df[f"{feature_name}_topic_assignment"] == topic) & ((df[text_column] != "") | pd.isna(df[text_column]))
        topic_msgs = df[topic_messages_filter]

        # get the topic vector of the topic
        topic_vector = topic_vectors[topic]
        
        def calculate_similarity(row: pd.Series, topic_vector: pd.Series, text_embeddings_column: str) -> float:
            """
            Calculate the cosine similarity between a text vector and a topic vector
            Parameters:
                row (pd.Series): the row containing the text vector
                topic_vector (pd.Series): the topic vector
                text_embeddings_column (str): the name of the column containing the text embeddings
            Returns:
                float: the cosine similarity between the text vector and the topic vector
            """
            message_vector = np.array(row[text_embeddings_column]).reshape(1, -1)  
            return cosine_similarity(message_vector, topic_vector)[0][0]               
        
        def calculate_similarity_with_structural_info(row: pd.Series, 
                                 topic_vector: pd.Series, 
                                 text_embeddings_column: str,
                                 structural_embedding_chat_map: pd.Series) -> float:
            """
            Calculate the cosine similarity between a text vector and a topic vector
            Parameters:
                row (pd.Series): the row containing the text vector
                topic_vector (pd.Series): the topic vector
                text_embeddings_column (str): the name of the column containing the text embeddings
                structural_embedding_chat_map(pd.Series): a pd.Series where the index represents chat IDs and values are the combined average message and structural embeddings of the chat        
            Returns:
                float: the cosine similarity between the text vector and the topic vector
            """
            structural_part = structural_embedding_chat_map.get(row["telegram_chat_id"]) # get the structural embedding of the messages chat
            message_vector = np.concatenate([row["message_vector"], structural_part]) # add the structural embedding of the chat to the message 
            message_vector = message_vector.reshape(1, -1) # reshape the message vector to fit the specifications of the cosine_similarity function
            assert message_vector.shape[1] == topic_vector.shape[1] # check if the combined structural message vector has the same shape as the topic vector
            return cosine_similarity(message_vector, topic_vector)[0][0]  

        # reshape the topic vector to fit the specifications of the cosine_similarity function
        topic_vector = topic_vector.reshape(1, -1)

        # calculate the cosine similarity between the message vectors and the topic vector
        if add_structural_info:
            topic_msgs["similarity"] = topic_msgs.apply(lambda x: calculate_similarity_with_structural_info(x, topic_vector, text_embeddings_column, structural_embedding_chat_map), axis=1)
            print_log("calculate_similarity_with_structural_info", "Add Structural Information to Message Embeddings & Compare to Topic Vector", f"Completed ✓")
        else:
            print("Here 2")
            topic_msgs["similarity"] = topic_msgs.apply(lambda x: calculate_similarity(x, topic_vector, text_embeddings_column), axis=1)
            print_log("calculate_similarity_with_structural_info", "Compare Message Vectors to Topic Vector", f"Completed ✓")

        # drop duplicates to avoid displaying the same message multiple times. We use the preprocessing column to identify duplicates, as the original message text might contain insignificant differences like whitespaces
        topic_msgs = topic_msgs.drop_duplicates(subset=[text_preprocessed_column])

        # sort the messages by similarity and display the most similar messages
        topic_msgs = topic_msgs.sort_values(by="similarity", ascending=False)

        top_messages = topic_msgs[text_column].head(n)   
        
        top_topic_messages[topic] = list(top_messages.values)
        
        print_log("get_representative_texts", "Find Representative Texts", f"Completed ✓")
        
    return top_topic_messages


def create_topic_visualisations(topic_model, embeddings, texts_aggregations):
    # Visualize topics
    #topic_model.visualize_topics().show()  

    print("Topic Map:")
    # UMAP dimensionality reduction
    from umap import UMAP
    import numpy as np
    docs = texts_aggregations.tolist()
    embeddings = np.vstack(embeddings)
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    
    # Visualize documents using UMAP embeddings
    topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings).show()

    print("Bar Chart, displaying the top 13 topics and top 20 words per topic:")
    # Visualize bar chart for top 13 topics and 20 words per topic
    topic_model.visualize_barchart(top_n_topics=13, n_words=20).show()

    print("Hierarchical Topics:")
    # Visualize hierarchical topics
    hierarchical_topics = topic_model.hierarchical_topics(texts_aggregations)
    topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics).show()
    
    
def create_topic_vectors(topic_model: BERTopic, chat_vectors: pd.Series) -> pd.Series:
    """ Create a topic vector for each topic by averaging the chat vectors assigned to the topic.

    Parameters:
        topic_model (BERTopic): the topic model
        chat_vectors (pd.Series): the chat vectors used to create the topic vectors

    Returns:
        pd.Series: _description_
    """
    
    # create a map with chat ids and their corresponding topics
    topics = topic_model.topics_
    chat_ids= chat_vectors.index
    id_topic_map = pd.Series(topics, index=chat_ids, name='Topic')

    # calculate the mean vector of the chat vectors assigned to each topic
    topic_vectors = {}
    
    # iterate over all topics (excluding the "Other" topic)
    topics = topic_model.get_topic_info()["Topic"]
    topics = topics.values[topics.values != -1]
    for topic in topics:
        topic_chat_vectors = chat_vectors[id_topic_map == topic]
        topic_vector = np.mean(np.vstack(topic_chat_vectors), axis=0)
        topic_vectors[topic] = topic_vector
        
    topic_vectors = pd.Series(topic_vectors, name="Mean_Vector")
    
    return topic_vectors

def compare_averages(metrics_model_1: Dict[str, float],
                     topics_model_1: List[int], 
                     propabilities_model_1: List[float], 
                     model_1: BERTopic, 
                     vectors_1: pd.Series,
                     metrics_model_2: Dict[str, float],
                     topics_model_2: List[int],
                     propabilities_model_2: List[float],
                     model_2: BERTopic,
                     vectors_2: pd.Series) -> tuple[List[int], List[float], BERTopic, Dict[str, float], pd.Series]:
    """Compare two BERTopic models based on evaluation metrics and select the one with more favourable evaluation scores.

    Args:
        metrics_model_1 (Dict[str, float]): A dictionary containing the evaluation metrics (Coherence Score, Silhouette Score, Davies-Bouldin Score, etc.) for the first model.
        topics_model_1 (List[int]): A list of topic assignments for the documents in the first model.
        propabilities_model_1 (List[float]): A list of probabilities associated with the topics for the first model.
        model_1 (BERTopic): The first BERTopic model object to be compared.
        vectors_1 (pd.Series): The chat vectors used to create the topic vectors for the first model.
        metrics_model_2 (Dict[str, float]): A dictionary containing the evaluation metrics for the second model.
        topics_model_2 (List[int]): A list of topic assignments for the documents in the second model.
        propabilities_model_2 (List[float]): A list of probabilities associated with the topics for the second model.
        model_2 (BERTopic): The second BERTopic model object to be compared.
        vectors_2 (pd.Series): The chat vectors used to create the topic vectors for the second model.

    Raises:
        ValueError: Raised if both models perform equally well.

    Returns:
        List[int]: The topics assigned by the more favourable of the two BERTopic models.
        List[float]: The probabilities for each document's topic assignment by the more favourable of the two BERTopic models.
        BERTopic: The more favourable of the two BERTopic models.
        Dict[str, float]: A dictionary containing the evaluation metrics for the more favourable of the two BERTopic models.
        pd.Series: The chat vectors used to create the topic vectors for the more favourable of the two BERTopic models.
    """

    score_model_1 = 0
    score_model_2 = 0
    
    if metrics_model_1["avg_coherence_scores"] < metrics_model_2["avg_coherence_scores"]:
        score_model_1 += 1
    else:
        score_model_2 += 1
        
    if metrics_model_1["avg_silhouette_scores"] > metrics_model_2["avg_silhouette_scores"]:
        score_model_1 += 1
    else:
        score_model_2 += 1
        
    if metrics_model_1["avg_davies_bouldin_scores"] < metrics_model_2["avg_davies_bouldin_scores"]:
        score_model_1 += 1
    else:
        score_model_2 += 1
        
    if metrics_model_1["avg_noise_counts"] < metrics_model_2["avg_noise_counts"]:
        score_model_1 += 1
    else:
        score_model_2 += 1
        
    if score_model_1 > score_model_2:
        print("Model 1 is better based on the evaluated metrics.")
        return topics_model_1, propabilities_model_1, model_1, metrics_model_1, vectors_1
    
    elif score_model_2 > score_model_1:
        print("Model 2 is better based on the evaluated metrics.")
        return topics_model_2, propabilities_model_2, model_2, metrics_model_2, vectors_2
    
    else:
        raise ValueError("Both models are equally good based on the evaluated metrics.")