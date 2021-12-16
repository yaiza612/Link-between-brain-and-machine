import cv2
import numpy as np
import cv2 as cv
import pandas as pd

img_shape = (100, 100, 3)
black = (0, 0, 0)


def rotate_point(center, point, degree):
    x_pos = point[0]
    y_pos = point[1]
    x_ref = center[0]
    y_ref = center[1]
    x_dist = x_pos - x_ref
    y_dist = y_pos - y_ref
    x_add = x_dist * np.cos(degree) - y_dist * np.sin(degree)
    y_add = x_dist * np.sin(degree) + y_dist * np.cos(degree)
    return x_ref + x_add, y_ref + y_add


def sample_random_circle():
    """
    :return: a circle with area between 1 and 50% of the image
    """
    unit_circle_radius = 1 / np.sqrt(np.pi)
    min_radius = np.sqrt(img_shape[0]/np.pi)
    max_radius = img_shape[0]/np.sqrt(2 * np.pi)
    random_radius = int(min_radius + np.random.random() * (max_radius - min_radius))
    x_origin = np.random.randint(random_radius, 100 - random_radius)
    y_origin = np.random.randint(random_radius, 100 - random_radius)
    canvas = 255 * np.ones(shape=img_shape, dtype=np.uint8)
    circle = cv.circle(canvas, (x_origin, y_origin), random_radius, black, -1)
    return circle, np.count_nonzero(circle == 0)


def sample_random_triangle():
    """

    :return: a triangle with area between 1 and 50% of the image
    """
    def rotate_triangle(triangle, rotation):
        center = ((triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3,
                  (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3)
        rotated_triangle = []
        for point in triangle:
            rotated_point = rotate_point(center, point, rotation)
            rotated_triangle.append(rotated_point)
        return np.array(rotated_triangle)

    canvas = 255 * np.ones(shape=img_shape, dtype=np.uint8)
    random_rotation = np.pi * 2 * np.random.random()
    rand_size = img_shape[0]/10 + np.random.random() * (img_shape[0]/np.sqrt(2) - img_shape[0]/10)
    unit_triangle = np.array([[0, 0], [2 / np.power(3, 0.25), 0], [1 / np.power(3, 0.25), np.power(3, 0.25)]])
    rand_triangle = unit_triangle * rand_size
    rotated_triangle = rotate_triangle(rand_triangle, random_rotation)
    rotated_triangle = np.array(rotated_triangle, dtype=np.int32)
    for _ in range(10000):
        translated_triangle = []
        x_center = np.random.randint(low=img_shape[0]/10, high=img_shape[0]-img_shape[0]/10)
        y_center = np.random.randint(low=img_shape[0] / 10, high=img_shape[0] - img_shape[0] / 10)
        for point in rotated_triangle:
            translated_triangle.append([point[0] + x_center, point[1] + y_center])
        translated_triangle = np.array(translated_triangle)
        flattened = np.ravel(translated_triangle)
        if np.all(np.logical_and((0 <= flattened), (flattened <= 100))):
            pts = np.array([translated_triangle])
            triangle = cv2.fillPoly(canvas, pts, color=black)
            return triangle, np.count_nonzero(triangle == 0)
    return sample_random_triangle()


def sample_random_square():
    """

    :return: a square with area between 1 and 50% of the image
    """
    def rotate_square(square, rotation):
        center = (square[0] + square[2]) / 2
        rotated_square = []
        for point in square:
            rotated_point = rotate_point(center, point, rotation)
            rotated_square.append(rotated_point)
        return np.array(rotated_square)

    canvas = 255 * np.ones(shape=img_shape, dtype=np.uint8)
    random_rotation = np.pi * 2 * np.random.random()
    rand_size = img_shape[0]/10 + np.random.random() * (img_shape[0]/np.sqrt(2) - img_shape[0]/10)
    unit_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    rand_square = unit_square * rand_size

    rotated_square = rotate_square(rand_square, random_rotation)
    rotated_square = np.array(rotated_square, dtype=np.int32)
    for _ in range(10000):
        translated_square = []
        x_center = np.random.randint(low=img_shape[0]/10, high=img_shape[0]-img_shape[0]/10)
        y_center = np.random.randint(low=img_shape[0] / 10, high=img_shape[0] - img_shape[0] / 10)
        for point in rotated_square:
            translated_square.append([point[0] + x_center, point[1] + y_center])
        translated_square = np.array(translated_square)
        flattened = np.ravel(translated_square)
        if np.all(np.logical_and((0 <= flattened), (flattened <= 100))):
            pts = np.array([translated_square])
            square = cv2.fillPoly(canvas, pts, color=black)
            return square, np.count_nonzero(square == 0)
    return sample_random_square()


def save_samples(num_samples, shape):
    data = []
    if shape == "circle":
        for _ in range(num_samples):
            data.append(sample_random_circle()[0][:, :, 0])
    elif shape == "square":
        for _ in range(num_samples):
            data.append(sample_random_square()[0][:, :, 0])
    else:
        for _ in range(num_samples):
            data.append(sample_random_triangle()[0][:, :, 0])
    data = np.stack(data)
    np.save("test_" + shape, data)


def create_chimaera():
    choice = np.random.randint(0, 3)
    if choice == 0:
        square, num_black_pixels_square = sample_random_square()
        for _ in range(1000):
            triangle, num_black_pixels_triangle = sample_random_triangle()
            smaller_area = np.min((num_black_pixels_triangle, num_black_pixels_square))
            bigger_area = np.max((num_black_pixels_triangle, num_black_pixels_square))
            chimaera = np.logical_not(np.logical_or(square == 0, triangle == 0)) * 255
            num_black_pixels_chimaera = np.count_nonzero(chimaera == 0)
            if bigger_area + 0.5 * smaller_area <= num_black_pixels_chimaera:
                # at least 50% of the smaller shape should be visible
                return chimaera, num_black_pixels_square, num_black_pixels_triangle, num_black_pixels_chimaera, choice
    elif choice == 1:
        square, num_black_pixels_square = sample_random_square()
        for _ in range(1000):
            circle, num_black_pixels_circle = sample_random_circle()
            smaller_area = np.min((num_black_pixels_circle, num_black_pixels_square))
            bigger_area = np.max((num_black_pixels_circle, num_black_pixels_square))
            chimaera = np.logical_not(np.logical_or(square == 0, circle == 0)) * 255
            num_black_pixels_chimaera = np.count_nonzero(chimaera == 0)
            if bigger_area + 0.5 * smaller_area <= num_black_pixels_chimaera:
                # at least 50% of the smaller shape should be visible
                return chimaera, num_black_pixels_square, num_black_pixels_circle, num_black_pixels_chimaera, choice
    else:
        circle, num_black_pixels_circle = sample_random_circle()
        for _ in range(1000):
            triangle, num_black_pixels_triangle = sample_random_triangle()
            smaller_area = np.min((num_black_pixels_circle, num_black_pixels_triangle))
            bigger_area = np.max((num_black_pixels_circle, num_black_pixels_triangle))
            chimaera = np.logical_not(np.logical_or(triangle == 0, circle == 0)) * 255
            num_black_pixels_chimaera = np.count_nonzero(chimaera == 0)
            if bigger_area + 0.5 * smaller_area <= num_black_pixels_chimaera:
                # at least 50% of the smaller shape should be visible
                return chimaera, num_black_pixels_circle, num_black_pixels_triangle, num_black_pixels_chimaera, choice
    print("fail")
    return create_chimaera()

# important ratio: (num_pixels_chimaera - num_pixels_first_shape) /(num_pixels_chimaera - num_pixels_second_shape)

#for shape in ["square", "circle", "triangle"]:
 #   save_samples(10000, shape)

chimaeras = []
pixels_first_shape = []
pixels_second_shape = []
pixels_chimaera = []
choices = []
for _ in range(10000):
    if _ % 100 == 0:
        print(_)
    chimaera, first_shape, second_shape, pixel_chimaera, choice = create_chimaera()
    chimaeras.append(chimaera)
    pixels_first_shape.append(first_shape)
    pixels_second_shape.append(second_shape)
    pixels_chimaera.append(pixel_chimaera)
    choices.append(choice)
dict = {'shape_1': pixels_first_shape, 'shape_2': pixels_second_shape, 'chimaera': pixels_chimaera, 'choice': choices}
df = pd.DataFrame(dict)
df.to_csv('info_chimaeras.csv')
chimaeras = np.stack(chimaeras)
np.save("test_chimaeras_v2", chimaeras)

img = 255 * np.ones(shape=img_shape, dtype=np.uint8)


# The objective is create three classes of symbols: triangle, square, circle with different
# sizes and also different rotations
circle, _ = sample_random_circle()
cv.imwrite('circle.png', circle)


square, _ = sample_random_square()
cv.imwrite('square.png', square)

triangle, _ = sample_random_triangle()
cv.imwrite('triangle.png', triangle)
# later on create a class of chimeras, wich will be the circles, squares and triangles mixing

chimaera = create_chimaera()[0]
cv.imwrite('chimaera.png', chimaera)


#Create random pixels images
def create_random_pixels():
    img_shape = (100, 100, 3)
    canvas = 255 * np.ones(shape=img_shape, dtype=np.uint8)
    # I want a random percentage of pixels
    random_p = np.square(1 + np.random.random() * (np.sqrt(50) - 1))
    random_black_pixels = np.random.binomial(n=1, p=1 - random_p / 100, size=(100, 100, 3))
    shape_random_pixels = random_black_pixels * canvas
    shape_random_pixels = shape_random_pixels[:, :, 0]
    shape_random_pixels = np.expand_dims(shape_random_pixels, axis=-1)
    return shape_random_pixels

