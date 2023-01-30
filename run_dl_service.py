import json
import time
import redis
import tensorflow as tf
from object_detection.utils import label_map_util
from deep_learning import predict_detection
from image_processing import correct_image
from helpers import base64_decode_image, base64_encode_image
from pyzbar.pyzbar import decode
from settings import *
from tensorflow.python.ops.numpy_ops import np_config


# connect to Redis server
db = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT, db=REDIS_DB,
)

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS_QR, use_display_name=True,
)

np_config.enable_numpy_behavior()


def classify_process():

    detection_model = tf.saved_model.load(PATH_TO_MODEL_QR)

    while True:

        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        images_list = []
        tensors = []

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode('utf-8'))
            h = q['h']
            w = q['w']
            channels = q['channels']
            image = base64_decode_image(
                q['image'], IMAGE_DTYPE,
                (h, w, channels),
            )
            images_list.append(image)
            image_tensor = tf.convert_to_tensor(image)
            tensors.append(image_tensor)

            # update the list of image IDs
            imageIDs.append(q['id'])

        batch = tf.stack(tensors, axis=0)

        # check to see if we need to process the batch
        if len(imageIDs) > 0:

            results = predict_detection(batch, detection_model)

            for (imageID, resultSet, image) in zip(imageIDs, results, images_list):

                outputs = []
                h, w, channels = image.shape

                for i in range(len(resultSet)):

                    bbx = resultSet['coordinates'][i]
                    y1, x1, y2, x2 = bbx

                    y1_pixel = int(h*y1)
                    x1_pixel = int(w*x1)
                    y2_pixel = int(h*y2)
                    x2_pixel = int(w*x2)

                    qr_crop = image[y1_pixel:y2_pixel, x1_pixel:x2_pixel, :]
                    rotated_image = correct_image(qr_crop)
                    rotated_image = rotated_image.copy(order='C')
                    rotated_image_b64 = base64_encode_image(rotated_image)

                    barcodes = decode(rotated_image)
                    qr_content = None
                    if barcodes:
                        qr_content = barcodes[0].data.decode('ascii')

                    output = {
                        'qr_content': qr_content,
                        'confidence': str(resultSet['confidence'][i].numpy()),
                        'coordinates': str(bbx.numpy()),
                        'corrected_image': [rotated_image.shape, rotated_image_b64]
                    }

                    outputs.append(output)
                db.set(imageID, json.dumps(outputs))

            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(SERVER_SLEEP)


if __name__ == '__main__':
    classify_process()
