# Original work Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modified work Copyright 2018 Weiyan Shi.
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
# ==============================================================================

from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
# from tensorflow.python.util.tf_export import tf_export


def _maybe_tensor_shape_from_tensor(shape):
    if isinstance(shape, ops.Tensor):
        return tensor_shape.as_shape(tensor_util.constant_value(shape))
    else:
        return shape


def _concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.
    This function takes a size specification, which can be an integer, a
    TensorShape, or a Tensor, and converts it into a concatenated Tensor
    (if static = False) or a list of integers (if static = True).
    Args:
        prefix: The prefix; usually the batch size (and/or time step size).
            (TensorShape, int, or Tensor.)
        suffix: TensorShape, int, or Tensor.
        static: If `True`, return a python list with possibly unknown dimensions.
            Otherwise return a `Tensor`.
    Returns:
        shape: the concatenation of prefix and suffix.
    Raises:
        ValueError: if `suffix` is not a scalar or vector (or TensorShape).
        ValueError: if prefix or suffix was `None` and asked for dynamic
            Tensors out.
    """
    if isinstance(prefix, ops.Tensor):
        p = prefix
        p_static = tensor_util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = array_ops.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                                             "but saw tensor: %s" % p)
    else:
        p = tensor_shape.as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
                 if p.is_fully_defined() else None)
    if isinstance(suffix, ops.Tensor):
        s = suffix
        s_static = tensor_util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = array_ops.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                                             "but saw tensor: %s" % s)
    else:
        s = tensor_shape.as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
                 if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s"
                                             % (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape


def _best_effort_input_batch_size(flat_input):
    """Get static input batch size if available, with fallback to the dynamic one.
    Args:
        flat_input: An iterable of time major input Tensors of shape
            `[max_time, batch_size, ...]`.
        All inputs should have compatible batch sizes.
    Returns:
        The batch size in Python integer if available, or a scalar Tensor otherwise.
    Raises:
        ValueError: if there is any input with an invalid shape.
    """
    for input_ in flat_input:
        shape = input_.shape
        if shape.ndims is None:
            continue
        if shape.ndims < 2:
            raise ValueError(
                    "Expected input tensor %s to have rank at least 2" % input_)
        batch_size = shape[1].value
        if batch_size is not None:
            return batch_size
    # Fallback to the dynamic batch size of the first input.
    return array_ops.shape(flat_input[0])[1]


def _transpose_batch_time(x):
    """Transposes the batch and time dimensions of a Tensor.
    If the input tensor has rank < 2 it returns the original tensor. Retains as
    much of the static shape information as possible.
    Args:
        x: A Tensor.
    Returns:
        x transposed along the first two dimensions.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        return x

    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(
            x, array_ops.concat(
                    ([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(
            tensor_shape.TensorShape([
                    x_static_shape[1].value, x_static_shape[0].value
            ]).concatenate(x_static_shape[2:]))
    return x_t


def _infer_state_dtype(explicit_dtype, state):
    """Infer the dtype of an RNN state.
    Args:
        explicit_dtype: explicitly declared dtype or None.
        state: RNN's hidden state. Must be a Tensor or a nested iterable containing
            Tensors.
    Returns:
        dtype: inferred dtype of hidden state.
    Raises:
        ValueError: if `state` has heterogeneous dtypes or is empty.
    """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_sequence(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError("Unable to infer dtype from empty state.")
        all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
        if not all_same:
            raise ValueError(
                    "State has tensors of different inferred_dtypes. Unable to infer a "
                    "single representative dtype.")
        return inferred_dtypes[0]
    else:
        return state.dtype

def dynamic_vae(cell,
                inputs,
                dec_input_embedding,
                dec_seq_lens,
                output_tokens,
                z_t_size,
                sequence_length=None,
                initial_state=None,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None,
                initial_prev_z=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.
    Performs fully dynamic unrolling of `inputs`.
    Example:
    ```python
    # create a BasicRNNCell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
    # defining initial state
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                                                         initial_state=initial_state,
                                                                         dtype=tf.float32)
    ```
    ```python
    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                                         inputs=data,
                                                                         dtype=tf.float32)
    ```
    Args:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.
            If `time_major == False` (default), this must be a `Tensor` of shape:
                `[batch_size, max_time, ...]`, or a nested tuple of such
                elements.
            If `time_major == True`, this must be a `Tensor` of shape:
                `[max_time, batch_size, ...]`, or a nested tuple of such
                elements.
            This may also be a (possibly nested) tuple of Tensors satisfying
            this property.    The first two dimensions must match across all the inputs,
            but otherwise the ranks and other shape components may differ.
            In this case, input to `cell` at each time-step will replicate the
            structure of these tuples, except for the time dimension (from which the
            time is taken).
            The input to `cell` at each time step will be a `Tensor` or (possibly
            nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
        sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
            Used to copy-through state and zero-out outputs when past a batch
            element's sequence length.    So it's more for correctness than performance.
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state and expected output.
            Required if initial_state is not provided or RNN state has a heterogeneous
            dtype.
        parallel_iterations: (Default: 32).    The number of iterations to run in
            parallel.    Those operations which do not have any temporal dependency
            and can be run in parallel, will be.    This parameter trades off
            time for space.    Values >> 1 use more memory but take less time,
            while smaller values use less memory but computations take longer.
        swap_memory: Transparently swap the tensors produced in forward inference
            but needed for back prop from GPU to CPU.    This allows training RNNs
            which would typically not fit on a single GPU, with very minimal (or no)
            performance penalty.
        time_major: The shape format of the `inputs` and `outputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.    However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        scope: VariableScope for the created subgraph; defaults to "rnn".
    Returns:
        A pair (outputs, state) where:
        outputs: The RNN output `Tensor`.
            If time_major == False (default), this will be a `Tensor` shaped:
                `[batch_size, max_time, cell.output_size]`.
            If time_major == True, this will be a `Tensor` shaped:
                `[max_time, batch_size, cell.output_size]`.
            Note, if `cell.output_size` is a (possibly nested) tuple of integers
            or `TensorShape` objects, then `outputs` will be a tuple having the
            same structure as `cell.output_size`, containing Tensors having shapes
            corresponding to the shape data in `cell.output_size`.
        state: The final state.    If `cell.state_size` is an int, this
            will be shaped `[batch_size, cell.state_size]`.    If it is a
            `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
            If it is a (possibly nested) tuple of ints or `TensorShape`, this will
            be a tuple having the corresponding shapes. If cells are `LSTMCells`
            `state` will be a tuple containing a `LSTMStateTuple` for each cell.
    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If inputs is None or an empty list.
    """
    #print(dec_input_embedding)
    with vs.variable_scope(scope or "VAE") as varscope:
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # By default, time_major==False and inputs are batch-major: shaped
        #     [batch, time, depth]
        # For internal calculations, we transpose to [time, batch, depth]
        flat_input = nest.flatten(inputs)
        flat_dec_input_embedding_1 = nest.flatten(dec_input_embedding[0])
        flat_dec_seq_len_1 = nest.flatten(dec_seq_lens[0])
        flat_output_tokens_1 = nest.flatten(output_tokens[0])
        flat_dec_input_embedding_2 = nest.flatten(dec_input_embedding[1])
        flat_dec_seq_len_2 = nest.flatten(dec_seq_lens[1])
        flat_output_tokens_2 = nest.flatten(output_tokens[1])

        if not time_major:
            # (B,T,D) => (T,B,D)
            flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
            flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

            flat_dec_input_embedding_1 = [ops.convert_to_tensor(dec_input_) for dec_input_ in flat_dec_input_embedding_1]
            flat_dec_input_embedding_1 = tuple(_transpose_batch_time(dec_input_) for dec_input_ in flat_dec_input_embedding_1)

            flat_dec_seq_len_1 = [ops.convert_to_tensor(dec_seq_len_) for dec_seq_len_ in flat_dec_seq_len_1]
            flat_dec_seq_len_1 = tuple(_transpose_batch_time(dec_seq_len_) for dec_seq_len_ in flat_dec_seq_len_1)

            flat_output_tokens_1 = [ops.convert_to_tensor(output_token_) for output_token_ in flat_output_tokens_1]
            flat_output_tokens_1 = tuple(_transpose_batch_time(output_token_) for output_token_ in flat_output_tokens_1)

            flat_dec_input_embedding_2 = [ops.convert_to_tensor(dec_input_) for dec_input_ in flat_dec_input_embedding_2]
            flat_dec_input_embedding_2 = tuple(_transpose_batch_time(dec_input_) for dec_input_ in flat_dec_input_embedding_2)

            flat_dec_seq_len_2 = [ops.convert_to_tensor(dec_seq_len_) for dec_seq_len_ in flat_dec_seq_len_2]
            flat_dec_seq_len_2 = tuple(_transpose_batch_time(dec_seq_len_) for dec_seq_len_ in flat_dec_seq_len_2)

            flat_output_tokens_2 = [ops.convert_to_tensor(output_token_) for output_token_ in flat_output_tokens_2]
            flat_output_tokens_2 = tuple(_transpose_batch_time(output_token_) for output_token_ in flat_output_tokens_2)


        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                        "sequence_length must be a vector of length batch_size, "
                        "but saw shape: %s" % sequence_length.get_shape())
            sequence_length = array_ops.identity(    # Just to find it in the graph.
                    sequence_length, name="sequence_length")

        batch_size = _best_effort_input_batch_size(flat_input)

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If there is no initial_state, you must give a dtype.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                    math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                    ["Expected shape for Tensor %s is " % x.name,
                     packed_shape, " but saw shape: ", x_shape])

        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)
        dec_input_embedding_1 = nest.pack_sequence_as(structure=dec_input_embedding[0], flat_sequence=flat_dec_input_embedding_1)
        dec_seq_len_1 = nest.pack_sequence_as(structure=dec_seq_lens[0], flat_sequence=flat_dec_seq_len_1)
        output_tokens_1 = nest.pack_sequence_as(structure=output_tokens[0], flat_sequence=flat_output_tokens_1)

        dec_input_embedding_2 = nest.pack_sequence_as(structure=dec_input_embedding[1], flat_sequence=flat_dec_input_embedding_2)
        dec_seq_len_2 = nest.pack_sequence_as(structure=dec_seq_lens[1], flat_sequence=flat_dec_seq_len_2)
        output_tokens_2 = nest.pack_sequence_as(structure=output_tokens[1], flat_sequence=flat_output_tokens_2)

        dec_input_embedding_1 = nest.pack_sequence_as(structure=dec_input_embedding[0], flat_sequence=flat_dec_input_embedding_1)
        dec_seq_len_1 = nest.pack_sequence_as(structure=dec_seq_lens[0], flat_sequence=flat_dec_seq_len_1)
        output_tokens_1 = nest.pack_sequence_as(structure=output_tokens[0], flat_sequence=flat_output_tokens_1)
        dec_input_embedding_2 = nest.pack_sequence_as(structure=dec_input_embedding[1], flat_sequence=flat_dec_input_embedding_2)
        dec_seq_len_2 = nest.pack_sequence_as(structure=dec_seq_lens[1], flat_sequence=flat_dec_seq_len_2)
        output_tokens_2 = nest.pack_sequence_as(structure=output_tokens[1], flat_sequence=flat_output_tokens_2)

        #print("here")
        #print(dec_input_embedding_1)
        dec_input_embedding = [dec_input_embedding_1, dec_input_embedding_2]
        dec_seq_lens = [dec_seq_len_1, dec_seq_len_2]
        output_tokens = [output_tokens_1, output_tokens_2]

        losses, z_ts, p_ts, bow_logits1, bow_logits2 = _dynamic_vae_loop(
                cell,
                inputs,
                dec_input_embedding,
                dec_seq_lens,
                output_tokens,
                z_t_size,
                state,
                parallel_iterations=parallel_iterations,
                swap_memory=swap_memory,
                sequence_length=sequence_length,
                dtype=dtype,
                initial_prev_z=initial_prev_z)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            losses = nest.map_structure(_transpose_batch_time, losses)
            z_ts = nest.map_structure(_transpose_batch_time, z_ts)
            p_ts = nest.map_structure(_transpose_batch_time, p_ts)
            bow_logits1 = nest.map_structure(_transpose_batch_time, bow_logits1)
            bow_logits2 = nest.map_structure(_transpose_batch_time, bow_logits2)

        return (losses, z_ts, p_ts, bow_logits1, bow_logits2)


def _dynamic_vae_loop(cell,
                      inputs,
                      dec_input_embedding,
                      dec_seq_lens,
                      output_tokens,
                      z_t_size,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None, initial_prev_z=None):
    """Internal implementation of Dynamic RNN.
    Args:
        cell: An instance of RNNCell.
        inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
            tuple of such elements.
        initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
            `cell.state_size` is a tuple, then this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        parallel_iterations: Positive Python int.
        swap_memory: A Python boolean
        sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
        dtype: (optional) Expected dtype of output. If not specified, inferred from
            initial_state.
    Returns:
        Tuple `(final_outputs, final_state)`.
        final_outputs:
            A `Tensor` of shape `[time, batch_size, cell.output_size]`.    If
            `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
            objects, then this returns a (possibly nested) tuple of Tensors matching
            the corresponding shapes.
        final_state:
            A `Tensor`, or possibly nested tuple of Tensors, matching in length
            and shapes to `initial_state`.
    Raises:
        ValueError: If the input depth cannot be inferred via shape inference
            from the inputs.
    """
    state = initial_state
    assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

    # state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_dec_input_embedding_1 = nest.flatten(dec_input_embedding[0])
    flat_dec_seq_len_1 = nest.flatten(dec_seq_lens[0])
    flat_output_tokens_1 = nest.flatten(output_tokens[0])
    flat_dec_input_embedding_2 = nest.flatten(dec_input_embedding[1])
    flat_dec_seq_len_2 = nest.flatten(dec_seq_lens[1])
    flat_output_tokens_2 = nest.flatten(output_tokens[1])
    # flat_output_size = nest.flatten(cell.output_size)

    loss_size = 1 # elbo_loss is scalara
    z_t_size = z_t_size # 10 states or 20 states
    p_t_size = z_t_size
    bow_logits_size1 = cell.vocab_size ## ******** if new variable
    bow_logits_size2 = cell.vocab_size ## ******** if new variable
    flat_loss_size = nest.flatten(loss_size)
    flat_zt_size = nest.flatten(z_t_size)
    flat_pt_size = nest.flatten(p_t_size)
    flat_bow_logits_size1 = nest.flatten(bow_logits_size1) ## ******** if new variable
    flat_bow_logits_size2 = nest.flatten(bow_logits_size2)

    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = _best_effort_input_batch_size(flat_input)

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                                                     for input_ in flat_input)
    dec_input_embedding_got_shape_1 = tuple(input_.get_shape().with_rank_at_least(4)
                                                     for input_ in flat_dec_input_embedding_1)
    dec_seq_lens_got_shape_1 = tuple(input_.get_shape().with_rank_at_least(2)
                                                     for input_ in flat_dec_seq_len_1)
    output_tokens_got_shape_1 = tuple(input_.get_shape().with_rank_at_least(3)
                                                     for input_ in flat_output_tokens_1)
    dec_input_embedding_got_shape_2 = tuple(input_.get_shape().with_rank_at_least(4)
                                                     for input_ in flat_dec_input_embedding_2)
    dec_seq_lens_got_shape_2 = tuple(input_.get_shape().with_rank_at_least(2)
                                                     for input_ in flat_dec_seq_len_2)
    output_tokens_got_shape_2 = tuple(input_.get_shape().with_rank_at_least(3)
                                                     for input_ in flat_output_tokens_2)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                    "Input size (depth of inputs) must be accessible via shape inference,"
                    " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                    "Time steps is not the same for all the elements in the input in a "
                    "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                    "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(
                array_ops.stack(size), _infer_state_dtype(dtype, state))

    #flat_zero_output = tuple(_create_zero_arrays(output)
    #                                                 for output in flat_output_size)
    #zero_output = nest.pack_sequence_as(structure=cell.output_size,
    #                                                                        flat_sequence=flat_zero_output)

    flat_zero_loss = tuple(_create_zero_arrays(loss) for loss in flat_loss_size)
    zero_loss = nest.pack_sequence_as(structure=loss_size,
                                      flat_sequence=flat_zero_loss)

    flat_zero_zt = tuple(_create_zero_arrays(z_t) for z_t in flat_zt_size)
    zero_zt = nest.pack_sequence_as(structure=z_t_size,
                                      flat_sequence=flat_zero_zt)

    flat_zero_pt = tuple(_create_zero_arrays(p_t) for p_t in flat_pt_size)
    zero_pt = nest.pack_sequence_as(structure=p_t_size,
                                      flat_sequence=flat_zero_pt)

    flat_zero_logits1 = tuple(_create_zero_arrays(bow_logits1) for bow_logits1 in flat_bow_logits_size1)
    zero_logits1 = nest.pack_sequence_as(structure=bow_logits_size1,
                                      flat_sequence=flat_zero_logits1)

    flat_zero_logits2 = tuple(_create_zero_arrays(bow_logits2) for bow_logits2 in flat_bow_logits_size2)
    zero_logits2 = nest.pack_sequence_as(structure=bow_logits_size2,
                                      flat_sequence=flat_zero_logits2)


    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)
    else:
        max_sequence_length = time_steps

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_vae") as scope:
        base_name = scope

    def _create_ta(name, element_shape, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            element_shape=element_shape,
                                            tensor_array_name=base_name + name)

    in_graph_mode = True
    if in_graph_mode:
        loss_ta = tuple(
                _create_ta(
                        "loss_%d" % i,
                        element_shape=(tensor_shape.TensorShape([const_batch_size]).concatenate(
                                                             _maybe_tensor_shape_from_tensor(tmp_loss_size))),
                        dtype=_infer_state_dtype(dtype, state))
                for i, tmp_loss_size in enumerate(flat_loss_size))
        zt_ta = tuple(
            _create_ta(
                "zt_%d" % i,
                element_shape=(tensor_shape.TensorShape([const_batch_size]).concatenate(
                    _maybe_tensor_shape_from_tensor(tmp_zt_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, tmp_zt_size in enumerate(flat_zt_size))
        pt_ta = tuple(
            _create_ta(
                "pt_%d" % i,
                element_shape=(tensor_shape.TensorShape([const_batch_size]).concatenate(
                    _maybe_tensor_shape_from_tensor(tmp_pt_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, tmp_pt_size in enumerate(flat_pt_size))
        bow_logit1_ta = tuple(
            _create_ta(
                "bow_logits1_t_%d" % i,
                element_shape=(tensor_shape.TensorShape([const_batch_size]).concatenate(
                    _maybe_tensor_shape_from_tensor(tmp_logit_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, tmp_logit_size in enumerate(flat_bow_logits_size1))
        bow_logit2_ta = tuple(
            _create_ta(
                "bow_logits2_t_%d" % i,
                element_shape=(tensor_shape.TensorShape([const_batch_size]).concatenate(
                    _maybe_tensor_shape_from_tensor(tmp_logit_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, tmp_logit_size in enumerate(flat_bow_logits_size2))


        input_ta = tuple(
                _create_ta(
                        "input_%d" % i,
                        element_shape=flat_input_i.shape[1:],
                        dtype=flat_input_i.dtype)
                for i, flat_input_i in enumerate(flat_input))
        input_ta = tuple(ta.unstack(input_)
                                         for ta, input_ in zip(input_ta, flat_input))

        dec_input_ta_1 = tuple(
            _create_ta(
                "dec_input_1_%d" % i,
                element_shape=flat_dec_input_i.shape[1:],
                dtype=flat_dec_input_i.dtype)
            for i, flat_dec_input_i in enumerate(flat_dec_input_embedding_1))
        dec_input_ta_1 = tuple(ta.unstack(dec_input_)
                         for ta, dec_input_ in zip(dec_input_ta_1, flat_dec_input_embedding_1))

        dec_seq_lens_ta_1 = tuple(
            _create_ta(
                "dec_seq_len_1_%d" % i,
                element_shape=flat_dec_seq_len_i.shape[1:],
                dtype=flat_dec_seq_len_i.dtype)
            for i, flat_dec_seq_len_i in enumerate(flat_dec_seq_len_1))
        dec_seq_lens_ta_1 = tuple(ta.unstack(dec_seq_lens_input_)
                         for ta, dec_seq_lens_input_ in zip(dec_seq_lens_ta_1, flat_dec_seq_len_1))

        output_token_ta_1 = tuple(
            _create_ta(
                "output_token_1_%d" % i,
                element_shape=flat_output_i.shape[1:],
                dtype=flat_output_i.dtype)
            for i, flat_output_i in enumerate(flat_output_tokens_1))
        output_token_ta_1 = tuple(ta.unstack(output_token_)
                         for ta, output_token_ in zip(output_token_ta_1, flat_output_tokens_1))

        dec_input_ta_2 = tuple(
            _create_ta(
                "dec_input_2_%d" % i,
                element_shape=flat_dec_input_i.shape[1:],
                dtype=flat_dec_input_i.dtype)
            for i, flat_dec_input_i in enumerate(flat_dec_input_embedding_2))
        dec_input_ta_2 = tuple(ta.unstack(dec_input_)
                         for ta, dec_input_ in zip(dec_input_ta_2, flat_dec_input_embedding_2))

        dec_seq_lens_ta_2 = tuple(
            _create_ta(
                "dec_seq_len_2_%d" % i,
                element_shape=flat_dec_seq_len_i.shape[1:],
                dtype=flat_dec_seq_len_i.dtype)
            for i, flat_dec_seq_len_i in enumerate(flat_dec_seq_len_2))
        dec_seq_lens_ta_2 = tuple(ta.unstack(dec_seq_lens_input_)
                         for ta, dec_seq_lens_input_ in zip(dec_seq_lens_ta_2, flat_dec_seq_len_2))

        output_token_ta_2 = tuple(
            _create_ta(
                "output_token_2_%d" % i,
                element_shape=flat_output_i.shape[1:],
                dtype=flat_output_i.dtype)
            for i, flat_output_i in enumerate(flat_output_tokens_2))
        output_token_ta_2 = tuple(ta.unstack(output_token_)
                         for ta, output_token_ in zip(output_token_ta_2, flat_output_tokens_2))

    else:
        output_ta = tuple([0 for _ in range(time_steps.numpy())]
                                            for i in range(len(flat_output_size)))
        input_ta = flat_input

    def _time_step(time, output_ta_t, z_ta_t, state, prev_z, p_ta_t, bow_logits1_ta_t, bow_logits2_ta_t):
        """Take a time step of the dynamic RNN.
        Args:
            time: int32 scalar Tensor.
            output_ta_t: List of `TensorArray`s that represent the output.
            state: nested tuple of vector tensors that represent the state.
        Returns:
            The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        if in_graph_mode:
            input_t = tuple(ta.read(time) for ta in input_ta)
            # prev_z_t_minus_1 = tuple(ta.read(time-1) for ta in zt_ta)
            # Restore some shape information
            for input_, shape in zip(input_t, inputs_got_shape):
                input_.set_shape(shape[1:])

            dec_input_t_1 = tuple(ta.read(time) for ta in dec_input_ta_1)
            # Restore some shape information
            for dec_input_, i_shape in zip(dec_input_t_1, dec_input_embedding_got_shape_1):
                dec_input_.set_shape(i_shape[1:])

            dec_seq_lens_t_1 = tuple(ta.read(time) for ta in dec_seq_lens_ta_1)
            # Restore some shape information
            for input_, i_shape in zip(dec_seq_lens_t_1, dec_seq_lens_got_shape_1):
                input_.set_shape(i_shape[1:])

            output_token_t_1 = tuple(ta.read(time) for ta in output_token_ta_1)
            # Restore some shape information
            for input_, i_shape in zip(output_token_t_1, output_tokens_got_shape_1):
                input_.set_shape(i_shape[1:])

            dec_input_t_2 = tuple(ta.read(time) for ta in dec_input_ta_2)
            # Restore some shape information
            for dec_input_, i_shape in zip(dec_input_t_2, dec_input_embedding_got_shape_2):
                dec_input_.set_shape(i_shape[1:])

            dec_seq_lens_t_2 = tuple(ta.read(time) for ta in dec_seq_lens_ta_2)
            # Restore some shape information
            for input_, i_shape in zip(dec_seq_lens_t_2, dec_seq_lens_got_shape_2):
                input_.set_shape(i_shape[1:])

            output_token_t_2 = tuple(ta.read(time) for ta in output_token_ta_2)
            # Restore some shape information
            for input_, i_shape in zip(output_token_t_2, output_tokens_got_shape_2):
                input_.set_shape(i_shape[1:])

        else:
            input_t = tuple(ta[time.numpy()] for ta in input_ta)

        print("before")
        print(dec_input_embedding[0])
        print(dec_input_t_1)
        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        dec_input_t_1 = nest.pack_sequence_as(structure=dec_input_embedding[0], flat_sequence=dec_input_t_1)
        dec_seq_lens_t_1 = nest.pack_sequence_as(structure=dec_seq_lens[0], flat_sequence=dec_seq_lens_t_1)
        output_token_t_1 = nest.pack_sequence_as(structure=output_tokens[0], flat_sequence=output_token_t_1)

        dec_input_t_2 = nest.pack_sequence_as(structure=dec_input_embedding[1], flat_sequence=dec_input_t_2)
        dec_seq_lens_t_2 = nest.pack_sequence_as(structure=dec_seq_lens[1], flat_sequence=dec_seq_lens_t_2)
        output_token_t_2 = nest.pack_sequence_as(structure=output_tokens[1], flat_sequence=output_token_t_2)

        #print("type-------")
        #print(type(dec_input_t_1))
        #print(len(dec_input_t_1))
        call_cell = lambda: cell(input_t, state, [dec_input_t_1, dec_input_t_2],
                                 [dec_seq_lens_t_1, dec_seq_lens_t_2], [output_token_t_1, output_token_t_2], forward=False,
                                 prev_z_t=prev_z)


        if sequence_length is not None:
            output, zts, new_state, pts, bow_logits1, bow_logits2 = _vae_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_loss,
                    zero_zt=zero_zt,
                    zero_pt=zero_pt,
                    zero_logits1=zero_logits1,
                    zero_logits2=zero_logits2,
                    state=state,
                    call_cell=call_cell,
                    #state_size=state_size,
                    skip_conditionals=True)
            zts_softmax = tf.nn.softmax(zts)
            zts_hard = tf.cast(tf.equal(zts_softmax, tf.reduce_max(zts_softmax, 1, keep_dims=True)), zts_softmax.dtype)
            zts_onehot = tf.stop_gradient(zts_hard - zts_softmax) + zts_softmax

            # print("zts_onehot")
            # print(zts_onehot)
            # print("state")
            # print(new_state)
            #import sys
            #sys.exit()
        else:
            (output, new_state) = call_cell()

        # Pack state if using state tuples
        output = nest.flatten(output)
        zts = nest.flatten(zts)
        pts = nest.flatten(pts)
        bow_logits1 = nest.flatten(bow_logits1)
        bow_logits2 = nest.flatten(bow_logits2)

        if in_graph_mode:
            output_ta_t = tuple(
                    ta.write(time, out) for ta, out in zip(output_ta_t, output))
            z_ta_t = tuple(
                ta.write(time, zt) for ta, zt in zip(z_ta_t, zts)
            )
            p_ta_t = tuple(
                ta.write(time, pt) for ta, pt in zip(p_ta_t, pts)
            )
            bow_logits1_ta_t = tuple(
                ta.write(time, logits1) for ta, logits1 in zip(bow_logits1_ta_t, bow_logits1)
            )
            bow_logits2_ta_t = tuple(
                ta.write(time, logits2) for ta, logits2 in zip(bow_logits2_ta_t, bow_logits2)
            )
        else:
            for ta, out in zip(output_ta_t, output):
                ta[time.numpy()] = out

        return (time + 1, output_ta_t, z_ta_t, new_state, zts_onehot, p_ta_t, bow_logits1_ta_t, bow_logits2_ta_t)

    if in_graph_mode:
        # Make sure that we run at least 1 step, if necessary, to ensure
        # the TensorArrays pick up the dynamic shape.
        loop_bound = math_ops.minimum(
                time_steps, math_ops.maximum(1, max_sequence_length))
    else:
        # Using max_sequence_length isn't currently supported in the Eager branch.
        loop_bound = time_steps

    prev_z = initial_prev_z
    _, loss_final_ta, z_t_final_ta, final_h_t, final_prev_z, p_t_final_ta, bow_logits1_final_ta, bow_logits2_final_ta = \
        control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, loss_ta, zt_ta, state, prev_z, pt_ta, bow_logit1_ta, bow_logit2_ta),
            parallel_iterations=parallel_iterations,
            #maximum_iterations=time_steps, # IMPORTANT change, not compatible in tf 1.0.1 and 1.3.0, this is the bug that caused headache in June
            swap_memory=swap_memory)

    # Unpack final output if not using output tuples.
    if in_graph_mode:
        final_losses = tuple(ta.stack() for ta in loss_final_ta)
        # Restore some shape information
        for output, output_size in zip(final_losses, flat_loss_size):
            out_shape = _concat(
                    [const_time_steps, const_batch_size], output_size, static=True)
            output.set_shape(out_shape)

        final_zts = tuple(ta.stack() for ta in z_t_final_ta)
        for cur_zt, cur_zt_size in zip(final_zts, flat_zt_size):
            zt_shape = _concat(
                    [const_time_steps, const_batch_size], cur_zt_size, static=True)
            cur_zt.set_shape(zt_shape)

        final_pts = tuple(ta.stack() for ta in p_t_final_ta)
        for cur_pt, cur_pt_size in zip(final_pts, flat_pt_size):
            pt_shape = _concat(
                    [const_time_steps, const_batch_size], cur_pt_size, static=True)
            cur_pt.set_shape(pt_shape)

        final_logits1 = tuple(ta.stack() for ta in bow_logits1_final_ta)
        for cur_logit1, cur_logit1_size in zip(final_logits1, flat_bow_logits_size1):
            logit1_shape = _concat(
                    [const_time_steps, const_batch_size], cur_logit1_size, static=True)
            cur_logit1.set_shape(logit1_shape)

        final_logits2 = tuple(ta.stack() for ta in bow_logits2_final_ta)
        for cur_logit2, cur_logit2_size in zip(final_logits2, flat_bow_logits_size2):
            logit2_shape = _concat(
                    [const_time_steps, const_batch_size], cur_logit2_size, static=True)
            cur_logit2.set_shape(logit2_shape)

    else:
        final_outputs = output_final_ta

    final_losses = nest.pack_sequence_as(
            structure=loss_size, flat_sequence=final_losses)
    final_zts = nest.pack_sequence_as(
        structure=z_t_size, flat_sequence=final_zts
    )
    final_pts = nest.pack_sequence_as(
        structure=p_t_size, flat_sequence=final_pts
    )
    final_logits1 = nest.pack_sequence_as(
        structure=bow_logits_size1, flat_sequence=final_logits1
    )
    final_logits2 = nest.pack_sequence_as(
        structure=bow_logits_size2, flat_sequence=final_logits2
    )
    if not in_graph_mode:
        final_losses = array_ops.stack(final_losses, axis=0)
        final_zts = array_ops.stack(final_zts, axis=0)

    return final_losses, final_zts, final_pts, final_logits1, final_logits2


def _vae_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, zero_zt, zero_pt, zero_logits1, zero_logits2, state, call_cell, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.
    Returns an (output, state) pair conditioned on `sequence_length`.
    When skip_conditionals=False, the pseudocode is something like:
    if t >= max_sequence_length:
        return (zero_output, state)
    if t < min_sequence_length:
        return call_cell()
    # Selectively output zeros or output, old state or new state depending
    # on whether we've finished calculating each row.
    new_output, new_state = call_cell()
    final_output = np.vstack([
        zero_output if time >= sequence_length[r] else new_output_r
        for r, new_output_r in enumerate(new_output)
    ])
    final_state = np.vstack([
        state[r] if time >= sequence_length[r] else new_state_r
        for r, new_state_r in enumerate(new_state)
    ])
    return (final_output, final_state)
    Args:
        time: int32 `Tensor` scalar.
        sequence_length: int32 `Tensor` vector of size [batch_size].
        min_sequence_length: int32 `Tensor` scalar, min of sequence_length.
        max_sequence_length: int32 `Tensor` scalar, max of sequence_length.
        zero_output: `Tensor` vector of shape [output_size].
        state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
            or a list/tuple of such tensors.
        call_cell: lambda returning tuple of (new_output, new_state) where
            new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
            new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
        state_size: The `cell.state_size` associated with the state.
        skip_conditionals: Python bool, whether to skip using the conditional
            calculations.    This is useful for `dynamic_rnn`, where the input tensor
            matches `max_sequence_length`, and using conditionals just slows
            everything down.
    Returns:
        A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
            final_output is a `Tensor` matrix of shape [batch_size, output_size]
            final_state is either a single `Tensor` matrix, or a tuple of such
                matrices (matching length and shapes of input `state`).
    Raises:
        ValueError: If the cell returns a state tuple whose length does not match
            that returned by `state_size`.
    """

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)
    flat_zero_zt = nest.flatten(zero_zt)
    flat_zero_pt = nest.flatten(zero_pt)
    flat_zero_logits1 = nest.flatten(zero_logits1) ## ******* add new variable
    flat_zero_logits2 = nest.flatten(zero_logits2) ## ******* add new variable

    # Vector describing which batch entries are finished.
    copy_cond = time >= sequence_length

    def _copy_one_through(output, new_output):
        # TensorArray and scalar get passed through.
        if isinstance(output, tensor_array_ops.TensorArray):
            return new_output
        if output.shape.ndims == 0:
            return new_output
        # Otherwise propagate the old or the new value.
        with ops.colocate_with(new_output):
            return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_zt, flat_new_state, flat_new_pt, flat_new_logits1, flat_new_logits2):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
                _copy_one_through(zero_output, new_output)
                for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_zt = [
                _copy_one_through(zero_zt, new_zt)
                for zero_zt, new_zt in zip(flat_zero_zt, flat_new_zt)]
        flat_new_pt = [
                _copy_one_through(zero_pt, new_pt)
                for zero_pt, new_pt in zip(flat_zero_pt, flat_new_pt)]
        flat_new_state = [
                _copy_one_through(state, new_state)
                for state, new_state in zip(flat_state, flat_new_state)]
        flat_new_logits1 = [          ## ********** add new variable
                _copy_one_through(zero_logits1, new_logits1)
                for zero_logits1, new_logits1 in zip(flat_zero_logits1, flat_new_logits1)]
        flat_new_logits2 = [
                _copy_one_through(zero_logits2, new_logits2)
                for zero_logits2, new_logits2 in zip(flat_zero_logits2, flat_new_logits2)]
        return flat_new_output + flat_new_zt + flat_new_state + flat_new_pt + flat_new_logits1 + flat_new_logits2

    def _maybe_copy_some_through():
        """Run RNN step.    Pass through either no or some past state."""
        new_output, new_zt, new_state, new_pt, new_bow_logits1, new_bow_logits2 = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_zt = nest.flatten(new_zt)
        flat_new_output = nest.flatten(new_output)
        flat_new_pt = nest.flatten(new_pt)
        flat_new_bow_logits1 = nest.flatten(new_bow_logits1) ## ********** add new variable
        flat_new_bow_logits2 = nest.flatten(new_bow_logits2) ## ********** add new variable

        return control_flow_ops.cond(
                # if t < min_seq_len: calculate and return everything
                time < min_sequence_length, lambda: flat_new_output + flat_new_zt + flat_new_state + flat_new_pt + flat_new_bow_logits1 + flat_new_bow_logits2,
                # else copy some of it through
                lambda: _copy_some_through(flat_new_output, flat_new_zt, flat_new_state, flat_new_pt, flat_new_bow_logits1, flat_new_bow_logits2))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.    We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.    This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_zt, new_state, new_pt, new_bow_logits1, new_bow_logits2 = call_cell()

        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_zt = nest.flatten(new_zt)
        new_output = nest.flatten(new_output)
        new_pt = nest.flatten(new_pt)
        new_bow_logits1 = nest.flatten(new_bow_logits1)
        new_bow_logits2 = nest.flatten(new_bow_logits2)
        final_output_and_state = _copy_some_through(new_output, new_zt, new_state, new_pt, new_bow_logits1, new_bow_logits2)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
                # if t >= max_seq_len: copy all state through, output zeros
                time >= max_sequence_length, empty_update,
                # otherwise calculation is required: copy some or all of it through
                _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_zero_zt) + len(flat_state) + len(flat_zero_pt) + len(flat_zero_logits1) + len(flat_zero_logits2):
        raise ValueError("Internal error: state and output were not concatenated "
                                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_zt = final_output_and_state[len(flat_zero_output):(len(flat_zero_output)+len(flat_zero_zt))]
    final_state = final_output_and_state[(len(flat_zero_output)+len(flat_zero_zt)):(len(flat_zero_output)+len(flat_zero_zt)+len(flat_state))]
    final_pt = final_output_and_state[(len(flat_zero_output)+len(flat_zero_zt)+len(flat_state)):(len(flat_zero_output)+len(flat_zero_zt)+len(flat_state)+len(flat_zero_pt))]
    final_logits1 = final_output_and_state[(len(flat_zero_output)+len(flat_zero_zt)+len(flat_state)+len(flat_zero_pt)):
                                           (len(flat_zero_output) + len(flat_zero_zt) + len(flat_state) + len(
                                               flat_zero_pt)+len(flat_zero_logits1))]
    final_logits2 = final_output_and_state[(len(flat_zero_output) + len(flat_zero_zt) + len(flat_state) + len(
                                               flat_zero_pt)+len(flat_zero_logits1)):] ## ******** add new variable
    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for zt, flat_zt in zip(final_zt, flat_zero_zt):
        zt.set_shape(flat_zt.get_shape())
    for pt, flat_pt in zip(final_pt, flat_zero_pt):
        pt.set_shape(flat_pt.get_shape())
    for logits1, flat_logits1 in zip(final_logits1, flat_zero_logits1):
        logits1.set_shape(flat_logits1.get_shape())
    for logits2, flat_logits2 in zip(final_logits2, flat_zero_logits2):
        logits2.set_shape(flat_logits2.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        if not isinstance(substate, tensor_array_ops.TensorArray):
            substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
            structure=zero_output, flat_sequence=final_output)
    final_zt = nest.pack_sequence_as(
            structure=zero_zt, flat_sequence=final_zt)
    final_pt = nest.pack_sequence_as(
            structure=zero_pt, flat_sequence=final_pt)
    final_state = nest.pack_sequence_as(
            structure=state, flat_sequence=final_state)
    final_logits1 = nest.pack_sequence_as(
            structure=zero_logits1, flat_sequence=final_logits1)
    final_logits2 = nest.pack_sequence_as(
            structure=zero_logits2, flat_sequence=final_logits2)

    return final_output, final_zt, final_state, final_pt, final_logits1, final_logits2
