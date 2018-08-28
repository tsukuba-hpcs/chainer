"""Collection of function implementations.

Functions are either implemented as :class:`~chainer.Function`\\ s or
:class:`~chainer.FunctionNode`\\ s.
"""

from chainer.functions.activation.clipped_relu import clipped_relu  # NOQA
from chainer.functions.activation.crelu import crelu  # NOQA
from chainer.functions.activation.elu import elu  # NOQA
from chainer.functions.activation.hard_sigmoid import hard_sigmoid  # NOQA
from chainer.functions.activation.leaky_relu import leaky_relu  # NOQA
from chainer.functions.activation.log_softmax import log_softmax  # NOQA
from chainer.functions.activation.lstm import lstm  # NOQA
from chainer.functions.activation.maxout import maxout  # NOQA
from chainer.functions.activation.prelu import prelu  # NOQA
from chainer.functions.activation.relu import relu  # NOQA
from chainer.functions.activation.rrelu import rrelu  # NOQA
from chainer.functions.activation.selu import selu  # NOQA
from chainer.functions.activation.sigmoid import sigmoid  # NOQA
from chainer.functions.activation.slstm import slstm  # NOQA
from chainer.functions.activation.softmax import softmax  # NOQA
from chainer.functions.activation.softplus import softplus  # NOQA
from chainer.functions.activation.swish import swish  # NOQA
from chainer.functions.activation.tanh import tanh  # NOQA
from chainer.functions.activation.tree_lstm import tree_lstm  # NOQA

from chainer.functions.array.broadcast import broadcast  # NOQA
from chainer.functions.array.broadcast import broadcast_to  # NOQA
from chainer.functions.array.cast import cast  # NOQA
from chainer.functions.array.concat import concat  # NOQA
from chainer.functions.array.copy import copy  # NOQA
from chainer.functions.array.depth2space import depth2space  # NOQA
from chainer.functions.array.diagonal import diagonal  # NOQA
from chainer.functions.array.dstack import dstack  # NOQA
from chainer.functions.array.expand_dims import expand_dims  # NOQA
from chainer.functions.array.flatten import flatten  # NOQA
from chainer.functions.array.flip import flip  # NOQA
from chainer.functions.array.fliplr import fliplr  # NOQA
from chainer.functions.array.flipud import flipud  # NOQA
from chainer.functions.array.get_item import get_item  # NOQA
from chainer.functions.array.hstack import hstack  # NOQA
from chainer.functions.array.im2col import im2col  # NOQA
from chainer.functions.array.moveaxis import moveaxis  # NOQA
from chainer.functions.array.pad import pad  # NOQA
from chainer.functions.array.pad_sequence import pad_sequence  # NOQA
from chainer.functions.array.permutate import permutate  # NOQA
from chainer.functions.array.repeat import repeat  # NOQA
from chainer.functions.array.reshape import reshape  # NOQA
from chainer.functions.array.resize_images import resize_images  # NOQA
from chainer.functions.array.rollaxis import rollaxis  # NOQA
from chainer.functions.array.scatter_add import scatter_add  # NOQA
from chainer.functions.array.select_item import select_item  # NOQA
from chainer.functions.array.separate import separate  # NOQA
from chainer.functions.array.space2depth import space2depth  # NOQA
from chainer.functions.array.spatial_transformer_grid import spatial_transformer_grid  # NOQA
from chainer.functions.array.spatial_transformer_sampler import spatial_transformer_sampler  # NOQA
from chainer.functions.array.split_axis import split_axis  # NOQA
from chainer.functions.array.squeeze import squeeze  # NOQA
from chainer.functions.array.stack import stack  # NOQA
from chainer.functions.array.swapaxes import swapaxes  # NOQA
from chainer.functions.array.tile import tile  # NOQA
from chainer.functions.array.transpose import transpose  # NOQA
from chainer.functions.array.transpose_sequence import transpose_sequence  # NOQA
from chainer.functions.array.vstack import vstack  # NOQA
from chainer.functions.array.where import where  # NOQA

from chainer.functions.connection.bilinear import bilinear  # NOQA
from chainer.functions.connection.convolution_2d import convolution_2d  # NOQA
from chainer.functions.connection.convolution_nd import convolution_nd  # NOQA
from chainer.functions.connection.deconvolution_2d import deconvolution_2d  # NOQA
from chainer.functions.connection.deconvolution_nd import deconvolution_nd  # NOQA
from chainer.functions.connection.deformable_convolution_2d_sampler import deformable_convolution_2d_sampler  # NOQA
from chainer.functions.connection.depthwise_convolution_2d import depthwise_convolution_2d  # NOQA
from chainer.functions.connection.dilated_convolution_2d import dilated_convolution_2d  # NOQA
from chainer.functions.connection.embed_id import embed_id  # NOQA
from chainer.functions.connection.linear import linear  # NOQA
from chainer.functions.connection.local_convolution_2d import local_convolution_2d  # NOQA
from chainer.functions.connection.n_step_gru import n_step_bigru  # NOQA
from chainer.functions.connection.n_step_gru import n_step_gru  # NOQA
from chainer.functions.connection.n_step_lstm import n_step_bilstm  # NOQA
from chainer.functions.connection.n_step_lstm import n_step_lstm  # NOQA
from chainer.functions.connection.n_step_rnn import n_step_birnn  # NOQA
from chainer.functions.connection.n_step_rnn import n_step_rnn  # NOQA
from chainer.functions.connection.shift import shift  # NOQA

from chainer.functions.evaluation.accuracy import accuracy  # NOQA
from chainer.functions.evaluation.binary_accuracy import binary_accuracy  # NOQA
from chainer.functions.evaluation.classification_summary import classification_summary  # NOQA
from chainer.functions.evaluation.classification_summary import f1_score  # NOQA
from chainer.functions.evaluation.classification_summary import precision  # NOQA
from chainer.functions.evaluation.classification_summary import recall  # NOQA
from chainer.functions.evaluation.r2_score import r2_score  # NOQA

from chainer.functions.loss.absolute_error import absolute_error  # NOQA
from chainer.functions.loss.black_out import black_out  # NOQA
from chainer.functions.loss.contrastive import contrastive  # NOQA
from chainer.functions.loss.crf1d import argmax_crf1d  # NOQA
from chainer.functions.loss.crf1d import crf1d  # NOQA
from chainer.functions.loss.cross_covariance import cross_covariance  # NOQA
from chainer.functions.loss.ctc import connectionist_temporal_classification  # NOQA
from chainer.functions.loss.decov import decov  # NOQA
from chainer.functions.loss.hinge import hinge  # NOQA
from chainer.functions.loss.huber_loss import huber_loss  # NOQA
from chainer.functions.loss.mean_absolute_error import mean_absolute_error  # NOQA
from chainer.functions.loss.mean_squared_error import mean_squared_error  # NOQA
from chainer.functions.loss.negative_sampling import negative_sampling  # NOQA
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy  # NOQA
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy  # NOQA
from chainer.functions.loss.squared_error import squared_error  # NOQA
from chainer.functions.loss.triplet import triplet  # NOQA
from chainer.functions.loss.vae import bernoulli_nll  # NOQA
from chainer.functions.loss.vae import gaussian_kl_divergence  # NOQA
from chainer.functions.loss.vae import gaussian_nll  # NOQA

from chainer.functions.math.average import average  # NOQA
from chainer.functions.math.basic_math import absolute  # NOQA
from chainer.functions.math.basic_math import add  # NOQA
from chainer.functions.math.batch_l2_norm_squared import batch_l2_norm_squared  # NOQA
from chainer.functions.math.bias import bias  # NOQA
from chainer.functions.math.ceil import ceil  # NOQA
from chainer.functions.math.clip import clip  # NOQA
from chainer.functions.math.cumprod import cumprod  # NOQA
from chainer.functions.math.cumsum import cumsum  # NOQA
from chainer.functions.math.det import batch_det  # NOQA
from chainer.functions.math.det import det  # NOQA
from chainer.functions.math.digamma import digamma  # NOQA
from chainer.functions.math.einsum import einsum  # NOQA
from chainer.functions.math.erf import erf  # NOQA
from chainer.functions.math.erfc import erfc  # NOQA
from chainer.functions.math.erfcinv import erfcinv  # NOQA
from chainer.functions.math.erfcx import erfcx  # NOQA
from chainer.functions.math.erfinv import erfinv  # NOQA
from chainer.functions.math.exponential import exp  # NOQA
from chainer.functions.math.exponential import log  # NOQA
from chainer.functions.math.exponential import log10  # NOQA
from chainer.functions.math.exponential import log2  # NOQA
from chainer.functions.math.exponential_m1 import expm1  # NOQA
from chainer.functions.math.fft import fft  # NOQA
from chainer.functions.math.fft import ifft  # NOQA
from chainer.functions.math.fix import fix  # NOQA
from chainer.functions.math.floor import floor  # NOQA
from chainer.functions.math.fmod import fmod  # NOQA
from chainer.functions.math.hyperbolic import cosh  # NOQA
from chainer.functions.math.hyperbolic import sinh  # NOQA
from chainer.functions.math.identity import identity  # NOQA
from chainer.functions.math.inv import batch_inv  # NOQA
from chainer.functions.math.inv import inv  # NOQA
from chainer.functions.math.lgamma import lgamma  # NOQA
from chainer.functions.math.linear_interpolate import linear_interpolate  # NOQA
from chainer.functions.math.log_ndtr import log_ndtr  # NOQA
from chainer.functions.math.logarithm_1p import log1p  # NOQA
from chainer.functions.math.logsumexp import logsumexp  # NOQA
from chainer.functions.math.matmul import batch_matmul  # NOQA
from chainer.functions.math.matmul import matmul  # NOQA
from chainer.functions.math.maximum import maximum  # NOQA
from chainer.functions.math.minimum import minimum  # NOQA
from chainer.functions.math.minmax import argmax  # NOQA
from chainer.functions.math.minmax import argmin  # NOQA
from chainer.functions.math.minmax import max  # NOQA
from chainer.functions.math.minmax import min  # NOQA
from chainer.functions.math.ndtr import ndtr  # NOQA
from chainer.functions.math.ndtri import ndtri  # NOQA
from chainer.functions.math.polygamma import polygamma  # NOQA
from chainer.functions.math.prod import prod  # NOQA
from chainer.functions.math.scale import scale  # NOQA
from chainer.functions.math.sign import sign  # NOQA
from chainer.functions.math.sparse_matmul import sparse_matmul  # NOQA
from chainer.functions.math.sqrt import rsqrt  # NOQA
from chainer.functions.math.sqrt import sqrt  # NOQA
from chainer.functions.math.square import square  # NOQA
from chainer.functions.math.squared_difference import squared_difference  # NOQA
from chainer.functions.math.sum import sum  # NOQA
from chainer.functions.math.sum import sum_to  # NOQA
from chainer.functions.math.tensordot import tensordot  # NOQA
from chainer.functions.math.trigonometric import arccos  # NOQA
from chainer.functions.math.trigonometric import arcsin  # NOQA
from chainer.functions.math.trigonometric import arctan  # NOQA
from chainer.functions.math.trigonometric import arctan2  # NOQA
from chainer.functions.math.trigonometric import cos  # NOQA
from chainer.functions.math.trigonometric import sin  # NOQA
from chainer.functions.math.trigonometric import tan  # NOQA

from chainer.functions.noise.dropout import dropout  # NOQA
from chainer.functions.noise.gaussian import gaussian  # NOQA
from chainer.functions.noise.gumbel_softmax import gumbel_softmax  # NOQA
from chainer.functions.noise.simplified_dropconnect import simplified_dropconnect  # NOQA
from chainer.functions.noise.zoneout import zoneout  # NOQA

from chainer.functions.normalization.batch_normalization import batch_normalization  # NOQA
from chainer.functions.normalization.batch_normalization import fixed_batch_normalization  # NOQA
from chainer.functions.normalization.batch_renormalization import batch_renormalization  # NOQA
from chainer.functions.normalization.batch_renormalization import fixed_batch_renormalization  # NOQA
from chainer.functions.normalization.group_normalization import group_normalization  # NOQA
from chainer.functions.normalization.l2_normalization import normalize  # NOQA
from chainer.functions.normalization.layer_normalization import layer_normalization  # NOQA
from chainer.functions.normalization.local_response_normalization import local_response_normalization  # NOQA

from chainer.functions.pooling.average_pooling_2d import average_pooling_2d  # NOQA
from chainer.functions.pooling.average_pooling_nd import average_pooling_nd  # NOQA
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d  # NOQA
from chainer.functions.pooling.max_pooling_nd import max_pooling_nd  # NOQA
# TODO(kmaehashi) this alias should be removed in
# https://github.com/chainer/chainer/pull/5198
from chainer.functions.pooling.roi_align_2d import roi_align_2d as roi_average_align_2d  # NOQA
from chainer.functions.pooling.roi_pooling_2d import roi_pooling_2d  # NOQA
from chainer.functions.pooling.spatial_pyramid_pooling_2d import spatial_pyramid_pooling_2d  # NOQA
from chainer.functions.pooling.unpooling_2d import unpooling_2d  # NOQA
from chainer.functions.pooling.unpooling_nd import unpooling_nd  # NOQA
from chainer.functions.pooling.upsampling_2d import upsampling_2d  # NOQA

from chainer.functions.util.forget import forget  # NOQA

# Aliases
from chainer.functions.math.average import average as mean  # NOQA
