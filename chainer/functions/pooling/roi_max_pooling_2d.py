# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work of _roi_pooling_slice, forward_cpu and backward_cpu:
# -----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------

# Original work of forward_gpu and backward_gpu:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# -----------------------------------------------------------------------------

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check

from chainer.functions.pooling.roi_pooling_2d import _roi_pooling_slice


def _pair(x):
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return x, x


class ROIMaxPooling2D(function.Function):

    """RoI max pooling over a set of 2d planes."""

    def __init__(self, outsize, spatial_scale):
        outh, outw = _pair(outsize)
        if not (isinstance(outh, int) and outh > 0):
            raise TypeError(
                'outsize[0] must be positive integer: {}, {}'
                .format(type(outh), outh))
        if not (isinstance(outw, int) and outw > 0):
            raise TypeError(
                'outsize[1] must be positive integer: {}, {}'
                .format(type(outw), outw))
        if isinstance(spatial_scale, int):
            spatial_scale = float(spatial_scale)
        if not (isinstance(spatial_scale, float) and spatial_scale > 0):
            raise TypeError(
                'spatial_scale must be a positive float number: {}, {}'
                .format(type(spatial_scale), spatial_scale))
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
            x_type.dtype == roi_type.dtype,
            roi_type.ndim == 2,
            roi_type.shape[1] == 4,
            roi_index_type.dtype == numpy.int32,
            roi_index_type.ndim == 1,
            roi_type.shape[0] == roi_index_type.shape[0],
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        # `numpy.zeros` needs to be used because the arrays can be
        # returned without having some of its values updated.
        top_data = numpy.zeros((n_rois, channels, self.outh, self.outw),
                               dtype=bottom_data.dtype)
        self.argmax_data = numpy.zeros(top_data.shape, numpy.int32)

        for i_roi in six.moves.range(n_rois):
            idx = bottom_roi_indices[i_roi]
            ymin, xmin, ymax, xmax = bottom_rois[i_roi]
            ymin = int(round(ymin * self.spatial_scale))
            xmin = int(round(xmin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            roi_height = max(ymax - ymin, 1)
            roi_width = max(xmax - xmin, 1)
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outh in six.moves.range(self.outh):
                sliceh, lenh = _roi_pooling_slice(
                    outh, strideh, height, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in six.moves.range(self.outw):
                    slicew, lenw = _roi_pooling_slice(
                        outw, stridew, width, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    roi_data = bottom_data[int(idx), :, sliceh, slicew]\
                        .reshape(channels, -1)
                    top_data[i_roi, :, outh, outw] =\
                        numpy.max(roi_data, axis=1)

                    # get the max idx respect to feature_maps coordinates
                    max_idx_slice = numpy.unravel_index(
                        numpy.argmax(roi_data, axis=1), (lenh, lenw))
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * width + max_idx_slice_w
                    self.argmax_data[i_roi, :, outh, outw] = max_idx_slice
        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=bottom_data.dtype)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.elementwise(
            '''
            raw T bottom_data, raw T bottom_rois, raw int32 bottom_roi_indices,
            T spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width
            ''',
            'T top_data, int32 argmax_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_roi_indices[n];
            int roi_start_h = round(bottom_rois[n * 4 + 0] * spatial_scale);
            int roi_start_w = round(bottom_rois[n * 4 + 1] * spatial_scale);
            int roi_end_h = round(bottom_rois[n * 4 + 2] * spatial_scale);
            int roi_end_w = round(bottom_rois[n * 4 + 3] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_height = max(roi_end_h - roi_start_h , 1);
            int roi_width = max(roi_end_w - roi_start_w, 1);
            float bin_size_h = static_cast<float>(roi_height)
                           / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                           / static_cast<float>(pooled_width);

            int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                          * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                          * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                        * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                        * bin_size_w));

            // Add roi offsets and clip to input boundaries
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            int maxidx = -1;
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int bottom_index = h * width + w;
                    if (bottom_data[data_offset + bottom_index] > maxval) {
                        maxval = bottom_data[data_offset + bottom_index];
                        maxidx = bottom_index;
                    }
                }
            }
            top_data = maxval;
            argmax_data = maxidx;
            ''', 'roi_max_pooling_2d_fwd'
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.outh, self.outw, top_data, self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = numpy.zeros(self._bottom_data_shape, bottom_rois.dtype)

        pooled_height = self.outh
        pooled_width = self.outw
        top_diff = gy[0]

        for i in six.moves.range(top_diff.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            c = int(i / pooled_width / pooled_height) % channels
            n = int(i / pooled_width / pooled_height / channels)

            roi_batch_ind = int(bottom_roi_indices[n])

            max_idx = self.argmax_data[n, c, ph, pw]
            h = int(max_idx / width)
            w = max_idx % width
            if max_idx != -1:
                bottom_diff[roi_batch_ind, c, h, w] += top_diff[
                    n, c, ph, pw]
        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        bottom_rois, bottom_roi_indices = inputs[1:]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(
            self._bottom_data_shape, bottom_rois.dtype)

        cuda.elementwise(
            '''
            raw T top_diff, raw int32 argmax_data,
            raw T bottom_rois, raw int32 bottom_roi_indices, int32 num_rois,
            T spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width
            ''',
            'raw T bottom_diff',
            '''
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int n = i / pooled_width / pooled_height / channels;

            int roi_batch_ind = bottom_roi_indices[n];
            int bottom_diff_offset =
                (roi_batch_ind * channels + c) * height * width;
            int top_diff_offset =
                (n * channels + c) * pooled_height * pooled_width;

            int max_index =
                argmax_data[top_diff_offset + ph * pooled_width + pw];
            if (max_index != -1) {
                atomicAdd(
                    &bottom_diff[bottom_diff_offset + max_index],
                    top_diff[top_diff_offset + ph * pooled_width + pw]);
            }
            ''', 'roi_max_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois, bottom_roi_indices,
          bottom_rois.shape[0], self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_diff, size=gy[0].size)

        return bottom_diff, None, None


def roi_max_pooling_2d(x, rois, roi_indices, outsize, spatial_scale):
    """Spatial Region of Interest (ROI) max pooling function.

    This function acts similarly to :func:`~chainer.functions.max_pooling_2d`,
    but it computes the maximum of input spatial patch for each channel with
    the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 4), and each datum is set as below:
            (y_min, x_min, y_max, x_max).
        roi_indices (~chainer.Variable): Input roi variable. The shape is
            expected to be (n: data size, ).
        outsize ((int, int) or int): Expected output size after pooled
            (height, width). ``outsize=o`` and ``outsize=(o, o)``
            are equivalent.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIMaxPooling2D(outsize, spatial_scale)(x, rois, roi_indices)
