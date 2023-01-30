import os

PATH_TO_LABELS_QR = os.getenv('PATH_TO_LABELS_QR')
PATH_TO_MODEL_QR = os.getenv('PATH_TO_MODEL_QR')
LOG_FILE = os.getenv('LOG_FILE')

# initialize Redis connection settings
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# data type
IMAGE_DTYPE = 'uint8'

# initialize constants used for server queuing
IMAGE_QUEUE = 'image_queue'
BATCH_SIZE = 16
SERVER_SLEEP = 0.3
CLIENT_SLEEP = 0.3
THRESHOLD_CONFIDENCE = 0.8