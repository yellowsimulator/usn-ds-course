import matplotlib.pyplot as plt
from data import get_train_test_data
from params import class_name_fashion_mnist


train_X, train_y, test_X, test_y = get_train_test_data()
plt.figure(figsize=(10,10))
for i in range(0,20):
    plt.subplot(5,5, i+1)
    plt.imshow(train_X[i] )
    label = train_y[i]
    plt.title(class_name_fashion_mnist[label])
    plt.xticks([])
plt.show()

