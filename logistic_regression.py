from utils import flatten, gather_dataset, normalize_data, show_random_img
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy

def train_model(training_x, training_y, penalty):
    solver = 'lbfgs'
    if penalty == 'l1':
        solver = 'liblinear'
    model = LogisticRegression(random_state=0, penalty=penalty, max_iter=10000, solver=solver).fit(training_x, training_y)
    return model    

training_x, training_y, num_training, img_dimension = gather_dataset('dataset/train_catvnoncat.h5', 'train')
testing_x, testing_y, num_testing, _ = gather_dataset('dataset/test_catvnoncat.h5', 'test')

# show_random_img(training_x)
# show_random_img(testing_x)

# os três canais das imagens são "fundidos" em um só
training_x_flat = flatten(training_x)
testing_x_flat = flatten(testing_x)

normalized_training_x = normalize_data(training_x_flat)
normalized_testing_x = normalize_data(testing_x_flat)

for penalty in [None, 'l1', 'l2']:
    model = train_model(normalized_training_x, training_y, penalty)

    training_pred_y = model.predict(normalized_training_x)
    print()
    print("===============================================")
    print("Regulaziração:", penalty)
    print("========= Estatísticas de treinamento =========")
    print("Acurácia:", numpy.mean(training_y == training_pred_y))
    print("Matriz de confusão:\n", confusion_matrix(training_y, training_pred_y, normalize='true'))

    testing_pred_y = model.predict(normalized_testing_x)
    print("========== Estatísticas de validação ==========")
    print("Acurácia: ", numpy.mean(testing_y == testing_pred_y))
    print("Matriz de confusão:\n", confusion_matrix(testing_y, testing_pred_y, normalize='true'))
    print("===============================================")
    print()