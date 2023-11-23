from utils import gather_dataset, flatten, normalize_data
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import Sequential, layers, optimizers
from numpy import ndarray
import matplotlib.pyplot as plt

def build_model(data_x: ndarray, activation, learning_rate):
    model = Sequential()
    ### camada de entrada
    model.add(layers.Flatten())
    
    ### camada escondida
    n_neurons = 60
    model.add(layers.Dense(n_neurons, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation=activation))
    
    ### camada de saída
    # classificação binária com um nodo de saída, então utiliza-se sigmoide
    model.add(layers.Dense(1, kernel_initializer="random_uniform", bias_initializer="random_uniform", activation="sigmoid"))

    ### método de treinamento
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy'])
    
    model.build(data_x.shape)
    return model

def plot_results(results, activation, learning_rate):
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label= 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label= 'Validation accuracy')
    plt.suptitle('Training and Validation accuracy')
    plt.title(f'Activation: {activation}; Learning Rate: {learning_rate}')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'b', label= 'Training loss')
    plt.plot(epochs, val_loss, 'r', label= 'Validation loss')
    plt.suptitle('Training and Validation loss')
    plt.title(f'Activation: {activation}; Learning Rate: {learning_rate}')
    plt.legend()
    plt.show()

training_x, training_y, num_training, img_dimension = gather_dataset('dataset/train_catvnoncat.h5', 'train')
testing_x, testing_y, num_testing, _ = gather_dataset('dataset/test_catvnoncat.h5', 'test')

# training_x_flat = flatten(training_x)
# testing_x_flat = flatten(testing_x)

normalized_training_x = normalize_data(training_x)
normalized_testing_x = normalize_data(testing_x)

# for activation in ['tanh', 'relu']: # sigmoid ruim
#     for learning_rate in [0.01, 0.05, 0.1]:
#         model = build_model(normalized_training_x, activation, learning_rate)
#         results = model.fit(normalized_training_x, training_y, validation_data=(normalized_testing_x, testing_y), batch_size=num_training, epochs=500)
#         plot_results(results, activation, learning_rate)

# melhor?
model = build_model(normalized_training_x, 'tanh', 0.005)
results = model.fit(normalized_training_x, training_y, validation_data=(normalized_testing_x, testing_y), batch_size=num_training, epochs=2000)
plot_results(results, 'tanh', 0.005)