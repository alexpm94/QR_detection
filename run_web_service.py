import io
import json
import logging
import time
import uuid

import flask
import redis
from flask_cors import CORS
from PIL import Image
from image_processing import prepare_image

from helpers import base64_encode_image
from settings import *

app = flask.Flask(__name__)
CORS(app)
db = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT, db=REDIS_DB,
)

logging.basicConfig(
    filename=LOG_FILE, level=logging.DEBUG,
)
logging.debug('Logger initialized')


@app.route('/')
def homepage():
    return 'Welcome to the Deep Learning REST API!'


@app.route('/test', methods=['GET'])
def test():
    return flask.jsonify({'Message': 'Service Running!'})


@app.route('/predict', methods=['POST'])
def predict():
    # Initialize response
    data = {'success': False}
    logging.info('Pointing to Predict')
    try:
        # Ensure an image was properly uploaded to our endpoint
        if flask.request.method == 'POST':
            logging.info('POST received')

            if flask.request.files.get('image'):
                logging.info('Image received, process will start')

                # Read the image in PIL format and prepare it detection
                image = flask.request.files['image'].read()
                image = Image.open(io.BytesIO(image))
                image_id = str(uuid.uuid4())

                image = prepare_image(image)

                IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS = image.shape

                # Ensure NumPy array is C-contiguous
                image = image.copy(order='C')

                # Construct queue data
                d = {
                    'id': image_id, 'h': IMAGE_HEIGHT, 'w': IMAGE_WIDTH, 'channels': IMAGE_CHANS,
                    'image': base64_encode_image(image,),
                }
                db.rpush(IMAGE_QUEUE, json.dumps(d))

                while True:

                    output = db.get(image_id)

                    if output is not None:

                        output = output.decode('utf-8')
                        data['predictions'] = json.loads(output)

                        db.delete(image_id)
                        break

                    time.sleep(CLIENT_SLEEP)

                data['success'] = True
    except Exception as e:
        logging.error(e)

    return flask.jsonify(data)


if __name__ == '__main__':
    print('* Starting web service...')
    app.run()
