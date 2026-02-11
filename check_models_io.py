import tensorflow as tf
import warnings; warnings.filterwarnings('ignore')

for model_name in ['face_landmark_ptq.tflite', 'iris_landmark_ptq.tflite']:
    try:
        interp = tf.lite.Interpreter(model_path=model_name)
        interp.allocate_tensors()
        print(f'\n=== {model_name} ===')
        for i, inp in enumerate(interp.get_input_details()):
            print(f"  Input {i}: shape={inp['shape']}, dtype={inp['dtype']}, name={inp['name']}")
        for i, out in enumerate(interp.get_output_details()):
            print(f"  Output {i}: shape={out['shape']}, dtype={out['dtype']}, name={out['name']}")
    except Exception as e:
        print(f'{model_name}: ERROR - {e}')
