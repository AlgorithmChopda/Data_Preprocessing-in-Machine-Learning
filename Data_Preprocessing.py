import cv2
from tensorflow.keras.utils import to_categorical
import numpy as np

class DataPreprocessing:

    def __init__(self, path, image_size):
        self.path = path
        self.image_size = image_size
        self.image_array = []

    def read_classes_from_directory(self):
        print('Reading Classes from the Directory....')
        self.Category = []

        for folder in os.listdir(self.path):
            self.Category.append(folder)

        print('Classes found : ', len(self.Category))
        self.classes_length = len(self.Category)
        print(self.Category)

    def read_images(self):
        print('Reading Images from the Directory....')
        img_array = []
        for category in self.Category:
            class_path = os.path.join(self.path, category)
            class_value = self.Category.index(category)
            i = 0;
            for image in os.listdir(class_path):
                # print(image)
                if i > 200:
                    break
                i = i + 1
                img_path = os.path.join(class_path, image)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                try:
                    temp = cv2.resize(img, (self.image_size, self.image_size))
                    img_array.append([temp, class_value])
                except:
                    print('exception')
                    pass
        print(len(img_array))
        return img_array

    def seperate_image_and_labels(self, img_array):
        images = []
        labels = []

        for img, lab in img_array:
            images.append(img)
            labels.append(lab)
        print(len(images))
        return images, labels

    def convert_to_numpy_array(self, images, labels):
        print('Converting to Numpy array......')
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)

        print(self.images.shape)

    def normalize_data(self):
        print('Normalizing data......')
        self.images = self.images / 255
        print(self.images.shape)

    def one_hot_encode_data(self):
        print('One hot Encode Data......')
        self.labels = to_categorical(self.labels, self.classes_length)

    def reshape_data(self):
        print('Reshaping Data......')
        self.images = self.images.reshape(-1, self.image_size, self.image_size, 1)
        print(self.images.shape)

    def load_data(self):
        self.read_classes_from_directory()
        img_array = self.read_images()
        images, labels = self.seperate_image_and_labels(img_array)
        self.convert_to_numpy_array(images, labels)
        self.normalize_data()
        self.one_hot_encode_data()
        self.reshape_data()

        print('Data successfully preprocessed')
        return self.images, self.labels


t_p = '/content/gdrive/MyDrive/Kaggle/training_set/training_set/'
te_p = '/content/gdrive/MyDrive/Kaggle/test_set/test_set/'

images = []
labels = []
obj = DataPreprocessing(t_p, 300)
images, labels = obj.load_data()