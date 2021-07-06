from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#Загрузка датасета
digits = datasets.load_digits()


#Показать случайные картинки
fig, axes = plt.subplots(4,4)
axes=axes.flatten()
for i, ax in enumerate(axes):
    dig_ind=np.random.randint(0,len(digits.images))
    ax.imshow(digits.images[dig_ind].reshape(8, 8))
    ax.set_title(digits.target[dig_ind])
#plt.show()

#Посчитать картинок какого класса сколько
dic={x:0 for x in range(10)}
for dig in digits.target:
    dic[dig]+=1
print(dic)


def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора
    :param data: np.array, данные (размер выборки, количество пикселей
    :return: data: np.array, данные (размер выборки, количество пикселей
    """

    return data - avg

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
    train_data = data[0:n1]; train_labels = labels[0:n1]
    validation_data = data[n1:n2]; validation_labels = labels[n1:n2]
    test_data = data[n2:n]; test_labels = labels[n2:n]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def softmax(vec):
    a = max(vec)
    den = np.sum(np.exp(vec - a))
    y = np.exp(vec - a)
    return y / den

class CorelationClassifier:

    def __init__(self, classes_count = 10):
        self.classes_count = classes_count
        self.weight = []
        self.bias = []
    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        if not len(self.weight):
            w = np.random.sample((self.classes_count, len(data[0])))*2 - 1
            b = np.random.sample(self.classes_count)
            self.weight = w
            self.bias = b
        new_w = np.zeros((self.classes_count, len(data[0])))
        new_b = np.zeros(self.classes_count)
        for i in range(0, len(data) - 1):
            t = np.zeros(10)
            t[labels[i]] = 1
            z = np.dot(self.weight, data[i]) + self.bias
            y = softmax(z)
            new_w += np.dot((y - t).reshape(-1, 1), data[i].reshape(1, -1))
            new_b += y - t
        new_w /= len(data)
        new_b /= len(data)
        self.weight -= 0.15 * new_w
        self.bias -= 0.15 * new_b
        #self.averages = []
        #pass
        #self.averages = np.array(self.averages)

    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        """
        res = []
        for i in range(0, len(data) - 1):
            z = np.dot(self.weight, data[i]) + self.bias
            y = softmax(z)
            res.append(y.argmax())
        return res

    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        okey = 0
        res = self.predict(data)
        for i in range(0, len(res)):
            if res[i] == labels[i]:
                okey += 1
        return okey / len(data)
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data, digits.target)
train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)

#Посчитать картинок какого класса сколько в обучающем датасете
dic={x:0 for x in range(10)}
for dig in train_labels:
    dic[dig]+=1
print(dic)


classifier=CorelationClassifier()
Valid = []
for i in range(0, 40):
    classifier.fit(train_data, train_labels)
    print(f"Training accuracy {classifier.accuracy(train_data, train_labels)}")
    print(f"Validation accuracy {classifier.accuracy(validation_data, validation_labels)}")
    Valid.append(classifier.accuracy(validation_data, validation_labels))
print(max(Valid))
print(f"Test accuracy {classifier.accuracy(test_data, test_labels)}")
