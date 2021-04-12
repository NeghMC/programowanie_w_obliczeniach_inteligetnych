from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from pandas import DataFrame
from itertools import product
import numpy as np
from os.path import exists, dirname
from os import mkdir

# for classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


def generate_textures_features(textures_folder="."):
    feature_names = ('dissimilarity', 'contrast', 'correlation', 'energy', 'homogeneity', 'ASM')  # cechy tekstur

    distances = (1, 3, 5)
    angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)  # odległości i kąty dla jakich mają być tworzone macierze

    def get_full_names():
        dist_str = ('1', '2', '5')
        angles_str = '0deg, 45deg, 90deg, 135deg'.split(',')
        return ['_'.join(f) for f in product(feature_names, dist_str, angles_str)]

    def get_glcm_feature_array(patch):
        patch_64 = (patch / np.max(patch) * 63).astype('uint8')  # dyskretyzacja obrazu do 64 poziomów jasności
        glcm = greycomatrix(patch_64, distances, angles, 64, True, True)
        feature_vector = []
        for feature in feature_names:
            feature_vector.extend(list(greycoprops(glcm, feature).flatten()))
        return feature_vector

    categories = ["wall", "door", "floor"]
    mkdir(textures_folder + "\\cropped")

    size = 128, 128  # rozmiar próbek

    features = []
    for category in categories:
        img = Image.open(textures_folder + "\\" + category + ".jpg")
        xr = np.random.randint(0, img.width - size[0], 10)
        yr = np.random.randint(0, img.height - size[1], 10)  # losowanie 10ciu położeń próbek ze zdjęcia
        for i, (x, y) in enumerate(zip(xr, yr)):  # enumerate() dodaje indeks
            img_sample = img.crop((x, y, x + size[0], y + size[1]))
            img_sample.save(textures_folder + "\\cropped\\" + category + f"{i:02d}" + ".jpg")
            img_grey = img.convert('L')  # konwersja do skali szarości
            feature_vector = get_glcm_feature_array(np.array(img_grey))  # generowanie cech tekstury
            feature_vector.append(category)
            features.append(feature_vector)

    full_feature_names = get_full_names()
    full_feature_names.append('Category')

    df = DataFrame(data=features, columns=full_feature_names)
    df.to_csv('textures_data.csv', sep=',', index=False)


def main():
    if not exists(r".\textures\cropped"):
        generate_textures_features(r".\textures")

    features = pd.read_csv('textures_data.csv', sep=',')

    data = np.array(features)
    x = (data[:, :-1]).astype('float64')
    y = data[:, -1]

    x_transform = PCA(n_components=3)
    xt = x_transform.fit_transform(x)

    red = y == 'wall'
    blue = y == 'door'
    green = y == 'floor'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xt[red, 0], xt[red, 1], xt[red, 2], c="r")
    ax.scatter(xt[blue, 0], xt[blue, 1], xt[blue, 2], c="b")
    ax.scatter(xt[green, 0], xt[green, 1], xt[green, 2], c="g")

    classifier = svm.SVC(gamma='auto')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    cm = confusion_matrix(y_test, y_pred, normalize='true')

    print(cm)

    disp = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
