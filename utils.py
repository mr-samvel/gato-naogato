from h5py import File
from random import randint
import matplotlib.pyplot as plot
import numpy

def gather_dataset(dataset_path, dataset_type):
    dataset = File(dataset_path, 'r')
    dataset_x = numpy.array(dataset[f'{dataset_type}_set_x'][:]) # array de imagens de shape (num_imgs, width, height, camadas RGB (3))
    dataset_y = numpy.array(dataset[f'{dataset_type}_set_y'][:]) # array de labels p/ as imagens (0: nao gato; 1: gato)
    num_imgs = dataset_x.shape[0]
    img_dimension = dataset_x.shape[1] # imagens são quadradas, shape[1] (comprimento) = shape[2] (altura)
    return dataset_x, dataset_y, num_imgs, img_dimension

def flatten(array: numpy.ndarray):
    return array.reshape(array.shape[0], -1)

def show_random_img(images: numpy.ndarray):
    idx = randint(0, images.shape[0]-1)
    plot.imshow(images[idx])
    plot.show()

def normalize_data(flat_data_x):
    # normaliza os valores dos pixels dividindo cada  um por 255 (rgb)
    # assim o valor de um pixel varia entre 0 e 1
    return flat_data_x/255