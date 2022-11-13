import tensorflow as tf


class TensorBoardLogger():
    
    def __init__(self):

        self.writers = {}

    def create_writer(self, writer_name: str, writer_path: str):
        self.writers[writer_name] = tf.summary.create_file_writer(writer_path)

    def write_scalar(self, writer_name: str, graph_name: str, scalar: float, step: int):
        writer = self.writers.get(writer_name)

        with writer.as_default():
            tf.summary.scalar(graph_name, scalar, step = step)

    def write_histogram(self, writer_name: str, histogram_name: str, data, step: int):
        writer = self.writers.get(writer_name)

        with writer.as_default():
            tf.summary.histogram(histogram_name, data, step = step)

    @property
    def writer_names(self):
        return list(self.writers.keys())