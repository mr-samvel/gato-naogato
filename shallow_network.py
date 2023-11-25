from utils import gather_dataset, flatten, normalize_data
from keras import Sequential, layers, optimizers, callbacks
from numpy import ndarray
import matplotlib.pyplot as plt

def build_model(data_x: ndarray, n_neurons, activation, learning_rate):
    model = Sequential()
    ### camada de entrada
    model.add(layers.Flatten())
    
    ### camada escondida
    model.add(layers.Dense(n_neurons, kernel_initializer='random_uniform', bias_initializer='random_uniform', activation=activation))
    
    ### camada de saída
    # classificação binária com um nodo de saída, então utiliza-se sigmoide
    model.add(layers.Dense(1, kernel_initializer="random_uniform", bias_initializer="random_uniform", activation="sigmoid"))

    ### método de treinamento
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, 'binary_crossentropy', ['accuracy'])
    
    model.build(data_x.shape)
    return model

def plot_results(results):
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(epochs, acc, 'b', label='Training accuracy')
    axes[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()

    axes[1].plot(epochs, loss, 'b', label='Training loss')
    axes[1].plot(epochs, val_loss, 'r', label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()

    plt.tight_layout()

    plt.show()

training_x, training_y, num_training, img_dimension = gather_dataset('dataset/train_catvnoncat.h5', 'train')
testing_x, testing_y, num_testing, _ = gather_dataset('dataset/test_catvnoncat.h5', 'test')

# training_x_flat = flatten(training_x)
# testing_x_flat = flatten(testing_x)

normalized_training_x = normalize_data(training_x)
normalized_testing_x = normalize_data(testing_x)

early_stopping = callbacks.EarlyStopping(patience=100, mode="min", start_from_epoch=800)

### geração de todas as combinações de modelos
# for activation in ['sigmoid', 'tanh', 'relu']:
#     for n_neurons in [60, 200, 1000, 4000]:
#         for learning_rate in [0.004, 0.008, 0.016, 0.06]:
#             model = build_model(normalized_training_x, n_neurons, activation, learning_rate)
#             results = model.fit(normalized_training_x, training_y, validation_data=(normalized_testing_x, testing_y), batch_size=num_training, epochs=5000, callbacks=[early_stopping])
#             plot_results(results, learning_rate)

### melhor resultado
model = build_model(normalized_training_x, 200, 'tanh', 0.008)
results = model.fit(normalized_training_x, training_y, validation_data=(normalized_testing_x, testing_y), batch_size=num_training, epochs=5000, callbacks=[early_stopping])
plot_results(results)