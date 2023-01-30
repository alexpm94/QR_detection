import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
from settings import THRESHOLD_CONFIDENCE

np_config.enable_numpy_behavior()


def run_inference(input_tensor, model):
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    # detection_classes should be ints.
    output_dict['num_detections'] = int(output_dict['num_detections'][0],)
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def predict_detection(batch, detection_model):
    # Actual detection.

    results_list = []
    output_dict = run_inference(batch, detection_model)

    for n in range(len(output_dict['detection_scores'])):
        index_scores = np.array(output_dict['detection_scores'][n] > THRESHOLD_CONFIDENCE).nonzero()[0]

        result = {
            'coordinates': output_dict['detection_boxes'][n][index_scores],
            'class_name': output_dict['detection_classes'][n][index_scores],
            'confidence': output_dict['detection_scores'][n][index_scores],
        }

        results_list.append(result)

    return results_list
