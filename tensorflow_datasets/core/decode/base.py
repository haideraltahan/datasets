# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image transformations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six
from tensorflow_datasets.core import api_utils
from tensorflow_datasets.core.utils import py_utils


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
  """Base decoder object.

  `tfds.decode.Decoder` allow to overwrite the default decoding,
  for instance to decode and crop at the same time, or to skip the image
  decoding entirely.

  All decoder should derive from this base class. The decoders have
  to the `self.feature` property corresponding to the feature to which the
  transformation is applied.

  To implement a decoder, the main method to overwrite is `decode_example`
  which take the serialized feature as input and return the decoded feature.

  If your decode method change the output dtype, you should also overwrite
  the `dtype` property. This is required for compatibility with
  `tfds.features.Sequence`.

  """

  def __init__(self):
    self.feature = None

  @api_utils.disallow_positional_args
  def setup(self, feature):
    """Transformation contructor.

    The initialization of decode object is deferred because the objects only
    know the builder/features on which it is used after it has been
    constructed, the initialization is done in this function.

    Args:
      feature: `tfds.features.FeatureConnector`, the feature to which is applied
        this transformation.

    """
    self.feature = feature

  @property
  def dtype(self):
    """Returns the `dtype` after decoding."""
    tensor_info = self.feature.get_tensor_info()
    return py_utils.map_nested(lambda t: t.dtype, tensor_info)

  @abc.abstractmethod
  def decode_example(self, serialized_example):
    """Decode the example feature field (eg: image).

    Args:
      serialized_example: `tf.Tensor` as decoded, the dtype/shape should be
        identical to `feature.get_serialized_info()`

    Returns:
      example: Decoded example.
    """
    raise NotImplementedError('Abstract class')


class SkipDecoding(Decoder):
  """Transformation which skip the decoding entirelly.

  Example of usage:

  ```python
  ds = ds.load(
      'imagenet2012',
      split='train',
      decoders={
          'image': tfds.decode.SkipDecoding(),
      }
  )

  for ex in ds.take(1):
    assert ex['image'].dtype == tf.string
  ```
  """

  @property
  def dtype(self):
    tensor_info = self.feature.get_serialized_info()
    return py_utils.map_nested(lambda t: t.dtype, tensor_info)

  def decode_example(self, serialized_example):
    """Forward the serialized feature field."""
    return serialized_example
