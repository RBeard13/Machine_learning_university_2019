# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import time
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle


def pickle_it(data, path):
    """
    Сохранить данные data в файл path
    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_it(path):
    """
    Достать данные из pickle файла
    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий и тестовый датасет
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    n = data.shape[0]
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    train_data, train_labels = data[0:n1], labels[0:n1]
    validation_data, validation_labels = data[n1:n2], labels[n1:n2]
    test_data, test_labels = data[n2:n], labels[n2:n]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

class TreeNode:
    divide_value = None
    split_characteristic = None

    left_child = None
    right_child = None

    is_leaf = False
    labels = None
    classes_count = None

    def __init__(self, divide_value, split_characteristic, left_child, right_child,
                 isLeaf=False, labels=None, classes_count=None):
        self.left_child = left_child
        self.right_child = right_child
        self.divide_value = divide_value
        self.split_characteristic = split_characteristic


class DesignTree:
    MAX_HEIGHT_TREE = None
    MIN_ENTROPY = None
    STEPS_COUNT = None
    FEATURES_SIZE = None
    FEATURES_TO_EVALUATE = None
    MIN_NODE_ELEMENTS = None
    CLASSES_COUNT=None

    tay = 0
    data = None
    labels = None
    root = None
    Nodes = 0

    def __init__(self, max_height_tree, min_entropy, steps_count, min_node_elements, tay):#, times_to_evaluate):
        self.MAX_HEIGHT_TREE = max_height_tree
        self.MIN_ENTROPY = min_entropy
        self.STEPS_COUNT = steps_count
        self.MIN_NODE_ELEMENTS = min_node_elements
        self.tay = tay
        #self.TIMES_TO_EVALUATE = times_to_evaluate

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        #self.VECTOR_SIZE = #TODO
        #self.CLASSES_COUNT = #TODO

        self.root = self.slowBuildTree(self.data, self.labels, 0)


    def getEntropy(self, labels):
        #Оценивает энтропию по labels
        gist_class = self.get_classes(labels)
        sum = 0
        for i in gist_class:
            if(gist_class[i]):
                p = gist_class[i]/len(labels)
                sum += p * math.log(p)
        return -1 * sum

    def get_classes(self, labels):
        #гистограмма классов содержащихся в labels
        gist_class = {i: 0 for i in range(self.CLASSES_COUNT)}
        for i in labels:
            gist_class[i] += 1
        return gist_class

    def getRandomSplitCharacteristics(self):
        #возвращает индексы features, по которым будет происходить поиск лучшего split
        return random.randrange(0, self.data.shape[1])
        pass
    def make_leaf_node(self, labels):

        list_node = TreeNode(None, None, None, None)
        list_node.is_leaf = True
        gist_class = self.get_classes(labels)
        p = []
        quantity_class = 0
        for i in gist_class:
            p.append(gist_class[i] / labels.size)
            if (gist_class[i]):
                quantity_class += 1
        list_node.classes_count = quantity_class
        list_node.labels = p
        # print(quantity_class, p)
        return list_node

    def slowBuildTree(self, data, labels, height_tree):
        self.Nodes += 1
        #print(self.Nodes)
        #print("\tHeight tree %s, length data %s" % (height_tree, len(data)))

        DataEntropy = self.getEntropy(labels)
        #print(DataEntropy)

        """
        Условия на создание терминального узла
        Вывод информации о каждом случае в консоль
        """
        if(height_tree >= self.MAX_HEIGHT_TREE or DataEntropy <= self.MIN_ENTROPY
                or len(labels) <= self.MIN_NODE_ELEMENTS):
            return self.make_leaf_node(labels)
        """
        Подсчёт лучшего разделения и поиск лучшего information gain
        """

        best_IG = [0, 0, 0]
        best_split = [0, 0, 0, 0]
        for i in range(self.tay):
            x = self.getRandomSplitCharacteristics()
            threshold_min = int(np.min(data[:, x]))
            threshold_max = int(np.max(data[:, x]))
            j = random.randint(threshold_min, threshold_max)
            #for j in range(threshold_min, threshold_max):
            left_node_data = []
            left_node_labels = []
            right_node_data = []
            right_node_labels = []
            for k in range(data.shape[0]):
                if data[k][x] > j:
                    right_node_data.append(data[k])
                    right_node_labels.append(labels[k])
                else:
                    left_node_data.append(data[k])
                    left_node_labels.append(labels[k])
            left_entropy = self.getEntropy(left_node_labels)
            right_entropy = self.getEntropy(right_node_labels)
            IG = self.getEntropy(labels) - (len(left_node_labels)/len(labels) * left_entropy \
                     + len(right_node_labels)/len(labels) * right_entropy)
            if IG > best_IG[0]:
                best_IG = IG, x, j
                best_split = left_node_data, left_node_labels, right_node_data, right_node_labels

        #print(best_IG)
        if(best_IG[0] == 0):
            return self.make_leaf_node(labels)

        """
        Деление выборки и перенаправление в новые узлы, рекурсивный вызов этой функции
        """
        left_child = self.slowBuildTree(np.array(best_split[0]), np.array(best_split[1]), height_tree + 1)
        right_child = self.slowBuildTree(np.array(best_split[2]), np.array(best_split[3]), height_tree + 1)
        internal_node = TreeNode(best_IG[2], best_IG[1], left_child, right_child)


        return internal_node


    def getPrediction(self, data):
        #Возвращает предсказание для новых данных на основе корня дерева
        prediction = []
        for i in data:
            temp_node = self.root
            split = True
            while split:
                if (temp_node.is_leaf):
                    prediction.append(temp_node.labels)
                    #prediction.append(temp_node.labels.index(max(temp_node.labels)))
                    split = False
                else:
                    if(i[temp_node.split_characteristic] > temp_node.divide_value):
                        temp_node = temp_node.right_child
                    else:
                        temp_node = temp_node.left_child
        prediction = np.array(prediction)
        return prediction
'''class RandomForest:

if __name__ == "__main__":

    """
    Загрузка датасета digits
    """

    """
    Формирование выборки
    """

    """
    Валидация по количеству случайных семплирований 5, 50, 250, 500, 1000, +500 в зависимости от мощности компьютера
    """

    """
    ДОП ЗАДАНИЕ
    Валидациия по максимальной высоте дерева - 1, 2, 3, 4, 5, 6, 7
    Валидация по максимальному количеству объектов в терминальном узле - 5, 10, 15, 20 или
    свои значения в зависимости от размера выборки
    """

    """
    Сохранение моделей, загрузка лучшей для тестирования
    """

    """
    Подсчет итоговой точности на тестовой выборке
    """
'''


start = time.time()
digits = datasets.load_digits()
train_data, train_labels, validation_data, validation_labels, test_data, test_labels\
    = train_val_test_split(digits.data, digits.target)

accuracy = []
pred = []
best_valid = [0, 0]
trees = 15
tay_sample = [125]
for tay in tay_sample:
    for number in range(trees):
        tree = DesignTree(7, 0.1, 2, 5, tay)
        tree.CLASSES_COUNT = 10
        tree.train(train_data, train_labels)
        prediction = tree.getPrediction(validation_data)
        if not(len(pred)):
            pred = prediction
        else:
            pred[:, :] += prediction[:, :]
        #print(pred)
    pred[:, :] = pred[:, :] / trees
    labels = np.argmax(pred, axis = 1)
    #print(labels)
    good_predict = 0
    for i, element in enumerate(validation_labels):
        if(element == labels[i]):
            good_predict += 1
    if(good_predict / len(validation_labels) > best_valid[0]):
        best_valid = good_predict / len(validation_labels), tay
#accuracy.append(good_predict / len(validation_labels))
    print(good_predict, good_predict / len(validation_labels), tay)


print("end build %s" %(time.time()-start))
#
# pickle_it(tree, "FIRST DATASET/tree_small_data_set.pickle")
# print 'end save tree'


# root = unpickle_it("FIRST DATASET/tree.pickle")
#
# good_predictions=0
# bad_predictions=0
# data=unpickle_it("FIRST DATASET/train_dataset.pickle")
# real_classes=unpickle_it("FIRST DATASET/train_classes.pickle")
# for i,element in enumerate(data):
#     probability, prCl = getPrediction(element, root)
#     if prCl==real_classes[i]:
#         good_predictions+=1
#     else:
#         bad_predictions+=1
# print("total accuracy: %s" % (float(good_predictions)/(good_predictions+bad_predictions)))





