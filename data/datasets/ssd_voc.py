"""Custom VOC Dataset wrapper for SSD

Ellis Brown
"""

import sys
from PIL import Image

from . import VOCDetection
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class SSDVOC(VOCDetection):
    """Wrapper for VOCDetection dataset that overrides the getitem function
    for training so that the data augmentation transforms can access the
    annotation
    """

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id).convert('RGB')
        width, height = img.size

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            if self.image_set == 'train':
                img, target = self.transform(img, target)
            else:
                img = self.transform(img)
        img.squeeze_(0)

        return img, target

#
# def precision_recall(num_gbboxes, num_detections, tp, fp, scores,
#                      dtype=tf.float64, scope=None):
#     """Compute precision and recall from scores, true positives and false
#     positives booleans arrays
#     """
#     # Input dictionaries: dict outputs as streaming metrics.
#     if isinstance(scores, dict):
#         d_precision = {}
#         d_recall = {}
#         for c in num_gbboxes.keys():
#             scope = 'precision_recall_%s' % c
#             p, r = precision_recall(num_gbboxes[c], num_detections[c],
#                                     tp[c], fp[c], scores[c],
#                                     dtype, scope)
#             d_precision[c] = p
#             d_recall[c] = r
#         return d_precision, d_recall
#
#     # Sort by score.
#     with tf.name_scope(scope, 'precision_recall',
#                        [num_gbboxes, num_detections, tp, fp, scores]):
#         # Sort detections by score.
#         scores, idxes = tf.nn.top_k(scores, k=num_detections, sorted=True)
#         tp = tf.gather(tp, idxes)
#         fp = tf.gather(fp, idxes)
#         # Computer recall and precision.
#         tp = tf.cumsum(tf.cast(tp, dtype), axis=0)
#         fp = tf.cumsum(tf.cast(fp, dtype), axis=0)
#         recall = _safe_div(tp, tf.cast(num_gbboxes, dtype), 'recall')
#         precision = _safe_div(tp, tp + fp, 'precision')
#         return tf.tuple([precision, recall])

# def streaming_tp_fp_arrays(num_gbboxes, tp, fp, scores,
#                            remove_zero_scores=True,
#                            metrics_collections=None,
#                            updates_collections=None,
#                            name=None):
#     """Streaming computation of True and False Positive arrays. This metrics
#     also keeps track of scores and number of grountruth objects.
#     """
#     # Input dictionaries: dict outputs as streaming metrics.
#     if isinstance(scores, dict) or isinstance(fp, dict):
#         d_values = {}
#         d_update_ops = {}
#         for c in num_gbboxes.keys():
#             scope = 'streaming_tp_fp_%s' % c
#             v, up = streaming_tp_fp_arrays(num_gbboxes[c], tp[c], fp[c], scores[c],
#                                            remove_zero_scores,
#                                            metrics_collections,
#                                            updates_collections,
#                                            name=scope)
#             d_values[c] = v
#             d_update_ops[c] = up
#         return d_values, d_update_ops
#
#     # Input Tensors...
#     with variable_scope.variable_scope(name, 'streaming_tp_fp',
#                                        [num_gbboxes, tp, fp, scores]):
#         num_gbboxes = math_ops.to_int64(num_gbboxes)
#         scores = math_ops.to_float(scores)
#         stype = tf.bool
#         tp = tf.cast(tp, stype)
#         fp = tf.cast(fp, stype)
#         # Reshape TP and FP tensors and clean away 0 class values.
#         scores = tf.reshape(scores, [-1])
#         tp = tf.reshape(tp, [-1])
#         fp = tf.reshape(fp, [-1])
#         # Remove TP and FP both false.
#         mask = tf.logical_or(tp, fp)
#         if remove_zero_scores:
#             rm_threshold = 1e-4
#             mask = tf.logical_and(mask, tf.greater(scores, rm_threshold))
#             scores = tf.boolean_mask(scores, mask)
#             tp = tf.boolean_mask(tp, mask)
#             fp = tf.boolean_mask(fp, mask)
#
#         # Local variables accumlating information over batches.
#         v_nobjects = _create_local('v_num_gbboxes', shape=[], dtype=tf.int64)
#         v_ndetections = _create_local('v_num_detections', shape=[], dtype=tf.int32)
#         v_scores = _create_local('v_scores', shape=[0, ])
#         v_tp = _create_local('v_tp', shape=[0, ], dtype=stype)
#         v_fp = _create_local('v_fp', shape=[0, ], dtype=stype)
#
#         # Update operations.
#         nobjects_op = state_ops.assign_add(v_nobjects,
#                                            tf.reduce_sum(num_gbboxes))
#         ndetections_op = state_ops.assign_add(v_ndetections,
#                                               tf.size(scores, out_type=tf.int32))
#         scores_op = state_ops.assign(v_scores, tf.concat([v_scores, scores], axis=0),
#                                      validate_shape=False)
#         tp_op = state_ops.assign(v_tp, tf.concat([v_tp, tp], axis=0),
#                                  validate_shape=False)
#         fp_op = state_ops.assign(v_fp, tf.concat([v_fp, fp], axis=0),
#                                  validate_shape=False)
#
#         # Value and update ops.
#         val = (v_nobjects, v_ndetections, v_tp, v_fp, v_scores)
#         with ops.control_dependencies([nobjects_op, ndetections_op,
#                                        scores_op, tp_op, fp_op]):
#             update_op = (nobjects_op, ndetections_op, tp_op, fp_op, scores_op)
#
#         if metrics_collections:
#             ops.add_to_collections(metrics_collections, val)
#         if updates_collections:
#             ops.add_to_collections(updates_collections, update_op)
#         return val, update_op

# def average_precision_voc07(precision, recall, name=None):
#     """Compute (interpolated) average precision from precision and recall Tensors.
#     The implementation follows Pascal 2007 guidelines.
#     See also: https://sanchom.wordpress.com/tag/average-precision/
#     """
#     with tf.name_scope(name, 'average_precision_voc07', [precision, recall]):
#         # Convert to float64 to decrease error on cumulated sums.
#         precision = tf.cast(precision, dtype=tf.float64)
#         recall = tf.cast(recall, dtype=tf.float64)
#         # Add zero-limit value to avoid any boundary problem...
#         precision = tf.concat([precision, [0.]], axis=0)
#         recall = tf.concat([recall, [np.inf]], axis=0)
#
#         # Split the integral into 10 bins.
#         l_aps = []
#         for t in np.arange(0., 1.1, 0.1):
#             mask = tf.greater_equal(recall, t)
#             v = tf.reduce_max(tf.boolean_mask(precision, mask))
#             l_aps.append(v / 11.)
#         ap = tf.add_n(l_aps)
#         return ap
