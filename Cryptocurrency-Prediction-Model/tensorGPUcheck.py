import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("GPU 디바이스 목록:")
print(tf.config.list_physical_devices('GPU'))