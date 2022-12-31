import tensorflow as tf
#import tensorflow.compat.v1 as tf #all the places where tf is im[ortwed use this line]
#tf.compat.v1.disable_eager_execution()

import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
import json


class KGEP(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':  # default
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        #rename to appH_indices
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        #rename to appT_indices
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        transD = open("../datas/TransD_Pykg2Vec.json", "r")
        data = json.load(transD)
        entity_embeddings = data['ent_transfer.weight']
        rel_embeddings = data['rel_transfer.weight']

        #rename to appH_matrix
        self.user_emb_matrix = tf.reshape(entity_embeddings, [-1, self.dim])  #same matrix different names
        self.entity_emb_matrix = tf.reshape(entity_embeddings, [-1, self.dim])
        self.relation_emb_matrix = tf.reshape(rel_embeddings, [-1, self.dim])

#        rename to appH_embeddings and appH_emnbeddings
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)

        entities, relations = self.get_neighbors(self.item_indices)
        #rename to appT_entities, appT_relations
        users, u_relationns = self.get_neighbors(self.user_indices)#REPEATING
#        print(u_relationns[0])

        #print
        self.r_interact = tf.nn.embedding_lookup(self.relation_emb_matrix, u_relationns[0])
        self.r_interact = tf.reduce_mean(self.r_interact, axis=1)#Taking mean of the embeddings
        # [batch_size, dim] 16 dimensions mean across dimensions

        self.item_embeddings_agg, self.aggregators = self.aggregate(entities, relations)#same
        self.user_embeddings_agg, self.u_aggregators = self.aggregate(users, u_relationns)
        for agg in self.u_aggregators:
            self.aggregators.append(agg)
        self.item_embeddings_cat = tf.concat([self.item_embeddings, self.item_embeddings_agg], axis=1)
        self.user_embeddings_cat = tf.concat([self.user_embeddings + self.r_interact, self.user_embeddings_agg], axis=1)
        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings_cat * self.item_embeddings_cat, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):  # item index
        seeds = tf.expand_dims(seeds, axis=1)  # （n, 1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):  # n_iter：iter times while computing item representation
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]),
                                           [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        """

        :param entities: items->entities
        :param relations: items->relations
        :return:
        """
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):  # (0->2)
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh) #doubt why act used not in last iteration
            else:
               aggregator = self.aggregator_class(self.batch_size, self.dim)# last has dropout layer FC network it is
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):  # (0->2) 1->3 1->5
                #    (0->2) (0->1)
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]#entity_evtors and relation_vectors contain neighbours
                vector = aggregator(self_vectors=entity_vectors[hop],  # items K-hop neighbors
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),#why hop+1 as it is neigbour
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),#bfs child parent child of child in child
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])
        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        # self.l2_loss = None
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
            # self.l2_loss = tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        precision = precision_score(labels, scores, average="weighted")
        recall = recall_score(labels, scores, average="weighted")

        return precision, recall

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
