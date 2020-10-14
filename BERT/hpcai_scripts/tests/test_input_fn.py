import tensorflow as tf
import os
import os.path as osp


def file_based_input_fn_builder(input_file,
                                batch_size,
                                seq_length,
                                is_training,
                                drop_remainder,
                                hvd=None):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d


file_list = [
    osp.join("/home/shenggui/HPCAI-2020/results", file_name)
    for file_name in os.listdir("/home/shenggui/HPCAI-2020/results")
]

d = file_based_input_fn_builder(input_file=file_list,
                                batch_size=32,
                                seq_length=128,
                                is_training=True,
                                drop_remainder=True)

print(d)
while True:
    continue