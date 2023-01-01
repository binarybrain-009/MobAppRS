# MobAppRS
Repository for Mobile based App Recommender System

The file starts at main.py
Sample_Extraction.py
It extracts the positive sample and creates false negative samples for the test set
1.Main.py
  -It imports train from train module
->Imports load_data from data_loaders module
Data_loaders.py
->load_data- Calls all the functions listed below to return the kg triples and adj matrices
->load_Samples- Loads all the positive negative samples file and calls the dataset_split to divide into train, test and valid
->dataset_split- Split the dataset into train test and valid
->load_kg-Loads the KG and then calls contruct_kg
->construct_kg- Construct the KG triplets dictionary
->Construct_Adj- Construct the adjacency entities and adj relations matrices adj entities has the entity number of the neighbour relations and the adj relation has the relation number of the attached relationship.
Train.py
->train-Takes the adj and kg file makes initializes the KGEP model, Calls the topk_Setting, does training in batches, gets the topk_Eval function and prints the precision recall and MAP-N
->topk_setting- Get all samples from train data and only positive samples from the test data, 
->get_feed_dict-Getting indices of app_Head and app_Tail and the entities.
->ctrl_eval-It calculates the Precision, Recall and Mean Average Precision
->precision-Calculate precision
->maps-Calculate mean average precision
->topk_eval-Evaluate our data, Calculate test_item_list, Get the user score against different test_item_list. Put it into item_score_map then sort it.
->get_user_record-All samples of training set and positive samples of test set are reserved

Model.py
Initializer-Initialize the KGEP objects
Build KGEP objects
-Get neighbours of entities and relations
-Take mean of relationship embeddings
-Aggregate embeddings of the neighbours
-Concatenate the embeddings
-Multiply the head and tail aggregated embeddings then normalize the score using sigmoid
Aggregator.py
Boilerplate code for several different Aggregator methods

