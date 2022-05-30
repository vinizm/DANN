import tensorflow as tf


class TensorBoardLogger():
    
    def __init__(self):

        self.writers = {}

    def create_writer(self, writer_name: str, writer_path: str):
        self.writers[writer_name] = tf.summary.create_file_writer(writer_path)

    def write(self, writer_name: str, graph_name: str, scalar: float, step: float):
        writer = self.writers.get(writer_name)

        with writer.as_default():
            tf.summary.scalar(graph_name, scalar, step = step)

    @property
    def writer_names(self):
        return list(self.writers.keys())