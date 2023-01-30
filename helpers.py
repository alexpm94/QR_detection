import base64
import numpy as np


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode('utf-8')


def base64_decode_image(image_b64, dtype, shape):
    """ Takes image string and return numpy array

    Args:
        image_b64(str): Image str
        dtype(np.dtype): Numpy dtype, could be np.uint8, np.float32, etc.
        shape(tuple): image shape

    Returns:
        image_array(np.array): Image in np array format

    """

    image_bytes = bytes(image_b64, encoding='utf-8')
    image_array = np.frombuffer(base64.decodebytes(image_bytes), dtype=dtype)
    image_array = image_array.reshape(shape)

    return image_array
