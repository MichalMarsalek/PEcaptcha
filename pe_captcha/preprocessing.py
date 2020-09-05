"""
Extracting individual characters from the whole picture. Works in the following way:
Detect contours. If there's exactly 5, create masks out of their insides.
Otherwise detect most significant colors and corresponding areas. Use median filter to reduce pixel noise.
Go tru each pair of areas detected by contours and colors.
For each such pair create a convex hull of these two masks and AND them together (= calculate intersection).
If there's a significant number of nonzero pixels, we have a character. Use it as a mask.
If at this point we have 3 or less characters, throw it away.
If we have exactly 4 split the widest in half.
Use the resulting 5 masks to get the individual characters and crop them.
Resize them all to be 24 pixels hight and padd so that each character is in the middle of a 24×24 image.

In terms of the functions definied here the pipeline goes like this:
reduce_colors --> split_to_chars_contour --> cut_finals
or
reduce_colors --> split_to_chars_contour --> cut_finals
              \-> split_to_chars_colors  -/
Author:
    Michal Maršálek
"""
import numpy as np
import cv2 as cv


def reduce_colors(image, clusters=16, rounds=1):
    """
    Reduces colors in the picture using the kmeans algorithm.

    Args:
        image: Input image.
        clusters (int): Number of colors - argument for the cv.kmeans function.
        rounds (int): Number of rounds - argument for the cv.kmeans function.

    Returns:
        Image with reduced colors.

    """
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv.kmeans(samples,
            clusters,
            None,
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
            rounds,
            cv.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape(image.shape)


def split_to_chars_contour(image):
    """
    Splits the image to characters based on continuos areas of nonwhite.

    Args:
        image: Input image.

    Yields:
        Mask for each character.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        im = np.zeros((image.shape[0], image.shape[1]))
        cv.fillPoly(im, pts=[c], color=255)
        yield im.astype(np.uint8)


def split_to_chars_colors(img):
    """
    Splits the image to characters based on continuos areas of the same color.

    Args:
        img: Input image.

    Yields:
        Mask for each character.
    """
    reduced = reduce_colors(img)
    all_rgb_codes = img.reshape(-1, img.shape[-1])
    colors, counts = np.unique(all_rgb_codes, axis=0, return_counts=True)
    colors = [x[0] for x in sorted(zip(colors, counts), key=lambda x:-x[1])]
    for color in colors[:10]:
        if list(color) in ([255,255,255], [254,254,254]):
            continue
        char = cv.inRange(img, color, color)
        blurred = cv.medianBlur(char, 3)
        contoursOfBlurred = cv.findContours(blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        hull = [cv.convexHull(c, False) for c in contoursOfBlurred]
        mask = np.zeros(char.shape, np.uint8)
        mask = cv.drawContours(mask, hull, -1, (255,255,255), -1)
        yield mask


def split_to_chars_combined(img):
    """
    Splits the image to characters based on combination of contour and color methods.
    Yields areas of significant overlaps of these two methods.

    Args:
        image: Input image.

    Yields:
        Mask for each character.
    """
    contours = list(split_to_chars_contour(img))
    colors = list(split_to_chars_colors(img))
    for color in colors:
        for contour in contours:
            charMaybe = cv.bitwise_and(color, contour)
            if cv.countNonZero(charMaybe) > 35:
                yield charMaybe


def convex_hull(img):
    """
    Calculates a convex hull of a mask.

    Args:
        img: Input mask.

    Returns:
        Convex hull of the input.
    """
    hull = cv.convexHull(cv.findNonZero(img), False)
    res = np.zeros(img.shape, np.uint8)
    return cv.drawContours(res, [hull], -1, (255,255,255), -1)


def split_to_chars_final(img):
    """
    Performs the locating and splitting of the characters in the image.

    Args:
        img: Input image.

    Returns:
        list of images, each containing one character.
        Empty list indicates that the process failed.
    """
    masks = list(split_to_chars_contour(img))
    if len(masks) != 5:
        masks = list(split_to_chars_combined(img))
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    masks = [convex_hull(cv.dilate(c, kernel)) for c in masks]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    prev = None
    finals = []
    for i,mask in enumerate(masks):
        final = cv.bitwise_and(thresh, mask)
        if prev is not None:
            final = cv.bitwise_and(final, 255-prev)
        prev = final
        finals.append(final)
    return cut_finals(list(cut_finals(finals)))


def cut_finals(finals):
    """
    Cuts the widest "character" in half.

    Args:
        finals: Separated parts of the original image.

    Returns:
        Zero or five images containing the separated characters.

    """
    finals.sort(key=lambda c: cv.boundingRect(c)[0])
    if len(finals) < 4:
        return []
    for i, c in enumerate(finals):
        (x, y, w, h) = cv.boundingRect(c)
        if len(finals) == 4 and i == max(range(4), key=lambda i: cv.boundingRect(finals[i])[2]):
            yield c[y:y+h,x:x+w//2]
            yield c[y:y+h,x+w//2:x+w]
        else:
            yield c[y:y+h,x:x+w]


def split_to_chars(img):
    """
    Locates and isolates characters in the captcha. Resizes each character to a standard (square) size.
    Args:
        img:

    Returns:

    """
    img = reduce_colors(img)
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=(255, 255, 255))
    chars = list(split_to_chars_final(img))[:5]
    for char in chars:
        w,h = char.shape
        char = cv.resize(char, (22*h//w,22))
        if char.shape[1] > char.shape[0]:
            char = char[0:22,0:22]
        char = cv.copyMakeBorder(char, 0, 0, (22-char.shape[1])//2, 22-char.shape[1]-(22-char.shape[1])//2, cv.BORDER_CONSTANT,value=0)
        char = cv.copyMakeBorder(char, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
        yield char


def get_raw_and_preprocessed(img):
    """
    Creates a "comparison" image of the original captcha and preprocessed characters.
    Args:
        img: Input image.

    Returns:
        Image - captcha at the top, isolated characters at the bottom.
    """
    pre = cv.hconcat(list(split_to_chars(img)))
    x = img.shape[1] - pre.shape[1]
    pre = cv.copyMakeBorder(pre, 0,0, x//2, x-x//2, cv.BORDER_CONSTANT,value=0)
    pre = cv.cvtColor(pre, cv.COLOR_GRAY2BGR)
    return cv.vconcat([img, pre])
