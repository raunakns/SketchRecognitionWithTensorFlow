import tensorflow as tf
from tensorflow_serving.session_bundle import exporter

export_path = '../../../../models'
print 'Exporting trained model to', export_path

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.generic_signature()