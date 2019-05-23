import tensorflow as tf

def get_shape(x, rank=None):
  """Returns the dimensions of a Tensor as list of integers or scale tensors.

  Args:
    x: N-d Tensor;
    rank: Rank of the Tensor. If None, will try to guess it.
  Returns:
    A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
      input tensor.  Dimensions that are statically known are python integers,
      otherwise they are integer scalar tensors.
  """
  if x.get_shape().is_fully_defined():
    return x.get_shape().as_list()
  else:
    static_shape = x.get_shape()
    if rank is None:
      static_shape = static_shape.as_list()
      rank = len(static_shape)
    else:
      static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]


def pad_axis(x, offset, size, axis=0, name=None):
  """Pad a tensor on an axis, with a given offset and output size.
  The tensor is padded with zero (i.e. CONSTANT mode). Note that the if the
  `size` is smaller than existing size + `offset`, the output tensor
  was the latter dimension.

  Args:
    x: Tensor to pad;
    offset: Offset to add on the dimension chosen;
    size: Final size of the dimension.
  Return:
    Padded tensor whose dimension on `axis` is `size`, or greater if
    the input vector was larger.
  """
  with tf.name_scope(name, 'pad_axis'):
    shape = get_shape(x)
    rank = len(shape)
    # Padding description.
    new_size = tf.maximum(size - offset - shape[axis], 0)
    pad1 = tf.stack([0] * axis + [offset] + [0] * (rank - axis - 1))
    pad2 = tf.stack([0] * axis + [new_size] + [0] * (rank - axis - 1))
    paddings = tf.stack([pad1, pad2], axis=1)
    x = tf.pad(x, paddings, mode='CONSTANT')
    # Reshape, to get fully defined shape if possible.
    # TODO: fix with tf.slice
    shape[axis] = size
    x = tf.reshape(x, tf.stack(shape))
    return x


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
  """Apply non-maximum selection to bounding boxes. In comparison to TF
  implementation, use classes information for matching.
  Should only be used on single-entries. Use batch version otherwise.

  Args:
    scores: N Tensor containing float scores.
    bboxes: N x 4 Tensor containing boxes coordinates.
    nms_threshold: Matching threshold in NMS algorithm;
    keep_top_k: Number of total object to keep after NMS.
  Return:
    classes, scores, bboxes Tensors, sorted by score.
      Padded with zero if necessary.
  """
  with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
    # Apply NMS algorithm.
    idxes = tf.image.non_max_suppression(bboxes, scores, keep_top_k, nms_threshold)
    scores = tf.gather(scores, idxes)
    bboxes = tf.gather(bboxes, idxes)
    # Pad results.
    scores = pad_axis(scores, 0, keep_top_k, axis=0)
    bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
    return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None, parallel=10):
  """Apply non-maximum selection to bounding boxes. In comparison to TF
  implementation, use classes information for matching.
  Use only on batched-inputs. Use zero-padding in order to batch output
  results.

  Args:
    scores: Batch x N Tensor/Dictionary containing float scores.
    bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
    nms_threshold: Matching threshold in NMS algorithm;
    keep_top_k: Number of total object to keep after NMS.
  Return:
    scores, bboxes Tensors/Dictionaries, sorted by score.
      Padded with zero if necessary.
  """
  # Dictionaries as inputs.
  if isinstance(scores, dict) or isinstance(bboxes, dict):
    with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
      d_scores = {}
      d_bboxes = {}
      for c in scores.keys():
        s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                nms_threshold=nms_threshold,
                                keep_top_k=keep_top_k)
        d_scores[c] = s
        d_bboxes[c] = b
      return d_scores, d_bboxes

  # Tensors inputs.
  with tf.name_scope(scope, 'bboxes_nms_batch'):
    r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1], nms_threshold, keep_top_k),
                  (scores, bboxes),
                  dtype=(scores.dtype, bboxes.dtype),
                  parallel_iterations=parallel,
                  back_prop=False,
                  swap_memory=False,
                  infer_shape=True)
    scores, bboxes = r
    return scores, bboxes

# if __name__ == '__main__':
#   import pickle
#
#   with open('../results.pickle', 'rb') as handle:
#     results = pickle.load(handle)
#   reg_pred = results['reg_pred']
#   cls_pred = results['cls_pred']
#
#   regs = tf.placeholder(tf.float32, [None, 8732, 4], 'reg')
#   clss = tf.placeholder(tf.float32, [None, 8732, 20], 'cls')
#
#   scores, bboxes = bboxes_nms_batch({i: clss[:, :, i] for i in range(20)},
#                                     {i: regs for i in range(20)})
#
#   with tf.Session() as sess:
#     s, b = sess.run([scores, bboxes], feed_dict={regs: reg_pred, clss: cls_pred[:, :, 1:]})
#
#     with open('results_nms.pickle', 'wb') as handle:
#       pickle.dump({'scores': s, 'bboxes': b}, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#   pass
