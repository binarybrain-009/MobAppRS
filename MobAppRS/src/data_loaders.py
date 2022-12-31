import numpy as np
import os


def load_data(args):
    #rename to n_appH and n_appT
    n_AppHead, n_AppTail, train_data, eval_data, test_data = load_samples(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)  # entity->relation->adjacency entity->Adjacency relation
    print('data loaded.')
    return n_AppHead, n_AppTail, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation

def load_samples(args):#user->app 0-users 1-apps
    print('reading positive negative file ...')
    sample_file = '../datas' + '/relation_exp1-test'
    if os.path.exists(sample_file + '.npy'):
        sample_np = np.load(sample_file + '.npy')
    else:
        sample_np = np.loadtxt(sample_file + '.txt', dtype=np.float)
        np.save(sample_file + '.npy', sample_np)

    #rename to n_appH and n_appT
    n_user = len(set(sample_np[:, 0]))#number of apps in head
    n_app = len(set(sample_np[:, 1]))#number of apps in tail
    train_data, eval_data, test_data = dataset_split(sample_np, args)

    temp = list()
    for i in train_data:
        if i[2]==1:
            temp.append(i[0])
            temp.append(i[1])
    temp = np.reshape(temp, [-1,2])
    np.savetxt("train.txt", temp, fmt="%d", delimiter='\t')
    temp2 = list()
    for i in test_data:
        if i[2] == 1:
            temp2.append(i[0])
            temp2.append(i[1])
    temp2 = np.reshape(temp2, [-1, 2])
    np.savetxt("test.txt", temp2, fmt="%d", delimiter='\t')
    return n_user, n_app, train_data, eval_data, test_data


def dataset_split(sample_np, args):
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_samples = sample_np.shape[0]

    eval_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * eval_ratio), replace=False)
    left = set(range(n_samples)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train_data = sample_np[train_indices]
    eval_data = sample_np[eval_indices]
    test_data = sample_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):  # entity->relation->adjacency entity->Adjacency relation
    # reading kg file
    kg_file = '../datas' + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 1]))  # sets of entities
    n_relation = len(set(kg_np[:, 2]))  # sets of relations

    kg = construct_kg(kg_np)  # construct knowledge graph
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)  # constructing adjacency matrix

    return n_entity, n_relation, adj_entity, adj_relation  # The first two items are the number of entities and relationships, and followed are adjacency matrix of entities and relationships. Each row represents K neighbors of each entity, and K is the specified parameter


def construct_kg(kg_np):
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        tail = triple[1]
        relation = triple[2]

        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))  # {head:[(tail,relation)...}
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))  # {tail:[(head,relation)...}
    return kg


def construct_adj(args, kg, entity_num):
    # each line of adj_entity stores the sampled neighbor entities for a given entity (n*m),n entities，m neighbor nodes
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])  # 格式为（tail/head, relation）
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
   # sp=0
    for a in adj_entity:
           print(a)
    return adj_entity, adj_relation