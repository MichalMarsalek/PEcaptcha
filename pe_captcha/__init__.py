from os.path import join
import pe_captcha.neural as neural
import pe_captcha.preprocessing as preprocessing
import keras.models
import numpy as np
import cv2 as cv
from os import path, makedirs
from os import listdir
import urllib.request


""" -------------------- Collecting data ------------------------ """


def download_captcha(url="https://projecteuler.net/captcha/show_captcha.php"):
    """
    Downloads a captcha from the internet.
    Args:
        url: Url adress of captcha.

    Returns:
        Downloaded image in OpenCV format.
    """
    with urllib.request.urlopen(url) as url_response:
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv.imdecode(img_array, -1)


def show_bigger(img, window="raw"):
    """
    Shows enlarged (2×, no interpolation) version of an image.

    Args:
        img: Image to be shown.
        window (str): Name of the window.
    """
    img = cv.resize(img, (img.shape[1]*2, img.shape[0]*2))
    cv.imshow("raw", img)


def show_huge(img, window="raw"):
    """
        Shows enlarged (5×, with interpolation) version of an image.

    Args:
        img: Image to be shown.
        window (str): Name of the window.
    """
    img = cv.resize(img, (img.shape[1]*5, img.shape[0]*5), interpolation=cv.INTER_NEAREST)
    cv.imshow(window, img)

def collect_training(model_name="model0", old_model_name=None):
    """
    Keeps downloading and showing you captchas to label.

    After each character is shown press corresponding key to save the character,
    Pressing "c" skips to next character, "b" skips to next captcha while "r" terminates the loop.
    If old model prediction is correct you can press the return key to save all 5 characters at once.

    Args:
        model_name (str): Model for which we are collecting.
        old_model_name: Old model used as collecting assistence.
    """
    old_model = load_model(old_model_name) if old_model_name else None
    while True:
        img = download_captcha()
        cv.namedWindow("raw")
        show_huge(img, "raw")
        pred, _, _ = solve(old_model, img) if old_model else None
        if pred:
            print(pred)
        characters = list(preprocessing.split_to_chars(img))
        for i, char in enumerate(characters):
            show_huge(char, "small")
            k = cv.waitKey(0)
            if pred and i == 0 and k == 13:
                for c,d in zip(pred, characters):
                    save_char(model_name, c, d)
                break
            if k == ord("b"):
                break
            if k == ord("r"):
                return
            if k == ord("c"):
                continue
            save_char(model_name, chr(k), char)

def save_char(model_name, label, img):
    """
    Saves a (label, image) pair to a training directory.

    Args:
        model_name (str): Name of the model for which we are collecting.
        label (str): Label for the character.
        img: Picture of the character.
    """
    i = 0
    dir = join(model_name, "training_data", label)
    if not path.exists(dir):
        makedirs(dir)
    while path.exists(join(dir, f"{i}.png")):
        i += 1
    cv.imwrite(join(dir, f"{i}.png"), img)


def collect_testing(model_name="model0", old_model_name = None):
    """
    Keeps downloading and showing you captchas to label.

    After a chaptcha is shown press corresponding keys to label it.
    Pressing "r" terminates the loop.
    If old model prediction is correct you can press the return key to save it.

    Args:
        model_name (str): Model for which we are collecting.
        old_model_name: Old model used as collecting assistence.
    """
    old_model = load_model(old_model_name) if old_model_name else None
    while True:
        img = download_captcha()
        pred, _, _ = solve(old_model, img) if old_model else None
        if pred:
            print(pred)
        cv.namedWindow("raw")
        show_bigger(img, "raw")
        label = ""
        while len(label) < 5:
            k = cv.waitKey(0)
            if k == 13 and pred:
                label = pred
                break
            if k == ord("r"):
                return
            label += chr(k)
        dir = join(model_name, "testing_data")
        if not path.exists(dir):
            makedirs(dir)
        cv.imwrite(join(dir, f"{label}.png"), img)


""" --------------- Building and using a model ------------------ """


def load_picture(img):
    #TODO: implement this, perhaps combine it with download_captcha function?
    raise NotImplementedError()


def build_and_train(model_name="model0"):
    """
    Builds, trains, and saves the neural network model.
    Args:
        model_name (str): Name of the function inside the neural module to be used.
            Also a directory pointing to the training data. The model will be saved here too.
    """
    model = getattr(neural, model_name)(join(model_name, "training_data"))
    model.save(join(model_name, "model"))


def load_model(model_name="model0"):
    """
    Loads a NN model.

    Args:
        model_name (str): Model to be loaded.

    Returns:
        Keras NN model.
    """
    return keras.models.load_model(join(model_name, "model"))


def solve(model, img):
    """
    Uses the preprocessing -> NN pipeline to solve a captcha.
    Args:
        model: NN model
        img: captcha in OpenCV2 format

    Returns:
        label (str): predicted text
        prob (float): probability of success (product of ind_prob)
        ind_prob Tuple[float, ...]: probabilities of success for individual characters

    """
    label = ""
    prob = 1
    ind_prob = []
    chars = preprocessing.split_to_chars(img)
    chars = [np.expand_dims(c, axis=2) for c in chars]
    prediction_vectors = model.predict(chars)
    for j in range(5):
        prediction_i = max(range(10), key=lambda i: prediction_vectors[j, i])
        prediction_prob = prediction_vectors[j, prediction_i]
        label += str(prediction_i)
        prob *= prediction_prob
        ind_prob.append(prediction_prob)
    ind_prob = tuple(ind_prob)
    return label, prob, ind_prob

""" ------------------- Testing a model ---------------------- """

def test_from_disk(model_name="model0"):
    """
    Calculates predictions for all testing data.

    Args:
        model_name(str): Model to be evaluated.

    Returns:
        (float): Succes rate.
    """
    model = load_model(model_name)
    dir = join(model_name, "testing_data")
    correct = 0
    total = 0
    for file in listdir(dir):
        image = cv.imread(f"{dir}\\{file}")
        total += 1
        try:
            pred, _, _ = solve(model, image)
            correct += (pred + ".png") == file
        except:
            continue
    return correct/total

def test_online(model_name="model0"):
    """
    Keeps downloading and solving captchas from the internet.
    Args:
        model_name: Model to be evaluated.
    """
    model = load_model(model_name)
    while True:
        img = download_captcha()
        show_bigger(preprocessing.get_raw_and_preprocessed(img), "raw")
        pred = solve(model, img)
        print(pred)
        k = chr(cv.waitKey())
        if k == "b":
            return









