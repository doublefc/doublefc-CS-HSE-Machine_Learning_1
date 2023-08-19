import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):

    
    len_ = len(feature_vector)
    #print(len_)
    sorted_ind = np.argsort(feature_vector)
    target_sorted = target_vector[sorted_ind]
    
    uni, ind_ = np.unique(feature_vector[sorted_ind], return_index=True)
    
    tresholds = (uni[1:]+uni[:-1])/2
    
    p_1_l = np.cumsum(target_sorted)[ind_[1:]-1]/(ind_[1:])
    p_0_l = 1 - p_1_l
    
    p_1_r = np.cumsum(target_sorted[::-1])[len_-ind_[1:]-1]/(len_-ind_[1:])
    p_0_r = 1 - p_1_r
    
    R_l = 1 - p_1_l**2 - p_0_l**2
    R_r = 1 - p_1_r**2 - p_0_r**2
    
    R = -ind_[1:] * R_l / len_ - (len_ - ind_[1:]) * R_r/len_
    best_ind = np.argmax(R)
    
    best_split = tresholds[best_ind]
    best_R = R[best_ind]
    
    return tresholds, R, best_split, best_R

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_sample_split = min_samples_split
        self._min_sample_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        
        if np.all(sub_y == sub_y[0]):
            
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
       
        if (self._max_depth != None):
            if depth > self._max_depth or (self._min_sample_split != None and len(sub_X) < self._min_sample_split):
                
                node["type"] = "terminal"
                node["class"] = sub_y[0]
                return                
        depth += 1
        
        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count 
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                #print(sorted_categories, categories_map)
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
                #print(feature_vector)
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            #print(4)
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        #print('sub', len(sub_X[split]))
        if (self._min_sample_leaf != None):
            if (len(sub_X[split]) < self._min_sample_leaf or len(sub_X[np.logical_not(split)]) < self._min_sample_leaf):
                node["type"] = "terminal"
                node["class"] = sub_y[0]
                return
        t = depth   
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], t)

    def _predict_node(self, x, node):
        
        if node['type'] == 'terminal':
            return node['class']

        feature_type = self._feature_types[node["feature_split"]]
        if feature_type == 'categorical':
            #print(x)
            if x[node['feature_split']] in node['categories_split']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif feature_type == 'real':
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 1)

    def predict(self, X):
        predicted = []
        for x in X:
            #print('x',x, X)
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_sample_split,
            'min_samples_leaf': self._min_sample_leaf
        }
