from google.cloud import storage
import sklearn.metrics
import numpy as np
import json
import io
import argparse
import pandas as pd
import os
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)

GOOGLE_CLOUD_PROJECT = 'my-project-987543321'  # @param
GOOGLE_APPLICATION_CREDENTIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'key.json')  # @param
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
os.environ['GOOGLE_CLOUD_PROJECT'] = GOOGLE_CLOUD_PROJECT


def json_serving_input_fn(feat_names):
    """
    Build the serving inputs

    Args:
      feat_name   - list, Name list of features used in the prediction model.

    Returns:
      tf.estimator.export.ServingInputReceive
    """

    def serving_input_fn():
        feat_cols = [tf.feature_column.numeric_column(x) for x in feat_names]
        inputs = {}
        for feat in feat_cols:
            inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    return serving_input_fn


def make_input_fn(data_file,
                  seq_len,
                  batch_size,
                  cols=None,
                  num_epochs=None,
                  shuffle=False,
                  train_flag=False,
                  filter_prob=1.0):
    """Input function for estimator.

    Input function for estimator.

    Args:
      data_file   - string, Path to input csv file.
      seq_len     - int, Length of time sequence.
      batch_size  - int, Mini-batch size.
      num_epochs  - int, Number of epochs
      shuffle     - bool, Whether to shuffle the data
      cols        - list, Columns to extract from csv file.
      train_flag  - bool, Whether in the training phase, in which we may
                    ignore sequences when all appliances are off.
      filter_prob - float, The probability to pass data sequences with all
                    appliances being 'off', only valid when train_flag is True.
    Returns:
      tf.data.Iterator.
    """

    def _mk_data(*argv):
        """Format data for further processing.

        This function slices data into subsequences, extracts the flags
        from the last time steps and treat each as the target for the subsequences.
        """
        data = {'ActivePower_{}'.format(i + 1): x
                for i, x in enumerate(tf.split(argv[0], seq_len))}
        # Only take the label of the last time step in the sequence as target
        flags = [tf.split(x, seq_len)[-1][0] for x in argv[1:]]
        return data, tf.cast(tf.stack(flags), dtype=tf.uint8)

    def _filter_data(data, labels):
        """Filter those sequences with all appliances 'off'.

        Filter those sequences with all appliances 'off'.
        However, with filter_prob we pass the sequence.
        """
        rand_num = tf.random_uniform([], 0, 1, dtype=tf.float64)
        thresh = tf.constant(filter_prob, dtype=tf.float64, shape=[])
        is_all_zero = tf.equal(tf.reduce_sum(labels), 0)
        return tf.logical_or(tf.logical_not(is_all_zero),
                             tf.less(rand_num, thresh))

    record_defaults = [tf.float64, ] + [tf.int32] * (len(cols) - 1)
    dataset = tf.contrib.data.CsvDataset([data_file, ],
                                         record_defaults,
                                         header=True,
                                         select_cols=cols)

    dataset = dataset.apply(
        tf.contrib.data.sliding_window_batch(window_size=seq_len))
    dataset = dataset.map(_mk_data)

    if train_flag:
        dataset = dataset.filter(_filter_data).shuffle(60 * 60 * 24 * 7)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def model_fn(features, labels, mode, params):
    """Build a customized model for energy disaggregation.

    The model authoring uses pure tensorflow.layers.

    Denote gross energy in the house as a sequence
    $(x_t, x_{t+1}, \cdots, x_{t+n-1}) \in \mathcal{R}^n$,
    and the on/off states of appliances at time $t$ as
    $y_{t} = \{y^i_t \mid y^i_t = [{appliance}\ i\ is\ on\ at\ time\ t ]\}$,
    then we are learning a function
    $f(x_t, x_{t+1}, \cdots, x_{t+n-1}) \mapsto \hat{y}_{t+n-1}$.

    Args:
      features: dict(str, tf.data.Dataset)
      labels: tf.data.Dataset
      mode: One of {tf.estimator.ModeKeys.EVAL,
                    tf.estimator.ModeKeys.TRAIN,
                    tf.estimator.ModeKeys.PREDICT}
      params: Other related parameters

    Returns:
      tf.estimator.EstimatorSpec.
    """

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info('TRAIN')
    else:
        tf.logging.info('EVAL | PREDICT')

    feat_cols = [tf.feature_column.numeric_column(
        x) for x in params['feat_cols']]
    seq_data = tf.feature_column.input_layer(features, feat_cols)

    if not params['use_keras']:
        tf.logging.info('Tensorflow authoring')

        seq_data_shape = tf.shape(seq_data)
        batch_size = seq_data_shape[0]

        # RNN network using multilayer LSTM
        cells = [tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(params['lstm_size']), input_keep_prob=1 - params['dropout_rate'])
            for _ in range(params['num_layers'])]
        lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

        # Initialize the state of each LSTM cell to zero
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        # Unroll multiple time steps and the output size is:
        # [batch_size, max_time, cell.output_size]
        outputs, states = tf.nn.dynamic_rnn(cell=lstm,
                                            inputs=tf.expand_dims(
                                                seq_data, -1),
                                            initial_state=state,
                                            dtype=tf.float32)

        # Flatten the 3D output to 2D as [batch_size, max_time * cell.output_size]
        flatten_outputs = tf.layers.Flatten()(outputs)

        # A fully connected layer. The number of output equals the number of target appliances
        logits = tf.layers.Dense(params['num_appliances'])(flatten_outputs)

    else:
        tf.logging.info('Keras authoring')

        # RNN network using multilayer LSTM with the help of Keras
        model = keras.Sequential()
        for _ in range(params['num_layers']):
            model.add(
                keras.layers.LSTM(params['lstm_size'],
                                  dropout=params['dropout_rate'],
                                  return_sequences=True)
            )

        # Flatten the 3D output to 2D as [batch_size, max_time * cell.output_size]
        model.add(keras.layers.Flatten())
        # A fully connected layer. The number of output equals the number of target appliances
        model.add(keras.layers.Dense(params['num_appliances']))

        # Logits can be easily computed using Keras functional API
        logits = model(tf.expand_dims(seq_data, -1))

    # Probability of turning-on of each appliances corresponding output are computed by applying a sigmoid function
    probs = tf.nn.sigmoid(logits)
    predictions = {
        'probabilities': probs,
        'logits': logits
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Binary cross entropy is used as loss function
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits)
    loss_avg = tf.reduce_mean(loss)

    predicted_classes = tf.cast(tf.round(probs), tf.uint8)
    precision = tf.metrics.precision(labels=labels,
                                     predictions=predicted_classes)
    recall = tf.metrics.recall(labels=labels,
                               predictions=predicted_classes)
    f1_score = tf.contrib.metrics.f1_score(labels=labels,
                                           predictions=predicted_classes)

    metrics = {'precision': precision,
               'recall': recall,
               'f_measure': f1_score}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss_avg, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op)


def run_experiment(params, path, SELECT_COLUMN):
    """Run the training and evaluate using the high level API

    Args:
      params: dict, dictionary of hyper-parameters related to the running experiment.
    """

    select_cols = SELECT_COLUMN
    feat_col_names = ['app_{}'.format(i + 1)
                      for i in range(params['seq_len'])]

    # Construct input function for training, evaluation and testing
    # Note: Don't filter on the evaluation and test data
    def train_input(): return make_input_fn(
        data_file=params['train_file'],
        seq_len=params['seq_len'],
        batch_size=params['train_batch_size'],
        cols=select_cols,
        train_flag=True,
        num_epochs=params['num_epochs'],
        filter_prob=params['filter_prob'])

    def eval_input(): return make_input_fn(
        data_file=params['eval_file'],
        seq_len=params['seq_len'],
        batch_size=params['eval_batch_size'],
        cols=select_cols,
        num_epochs=1)

    model_dir = os.path.join(
        os.path.join(os.path.dirname(
            os.path.abspath(__file__)), params['job_dir']),
        json.loads(os.environ.get('TF_CONFIG', '{}'))
        .get('task', {}).get('trial', '')
    )

    tf.logging.info('model dir {}'.format(model_dir))

    # Experiment running configuration
    # Checkpoint is configured to be saved every ten minutes
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=10000)
    run_config = run_config.replace(model_dir=model_dir)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       config=run_config,
                                       params=params)

    # Set training spec
    early_stopping = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator,
        metric_name='f_measure',
        max_steps_without_increase=200,
        min_steps=100,
        run_every_secs=10)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input,
                                        max_steps=params['train_steps'])
    # hooks=[early_stopping])

    # Set serving function, exporter and evaluation spec
    # The serving function is only applicable for JSON format input
    serving_function = json_serving_input_fn(feat_names=feat_col_names)
    exporter = tf.estimator.FinalExporter(name=params['model_name'],
                                          serving_input_receiver_fn=serving_function)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input,
                                      steps=None,
                                      throttle_secs=120,
                                      exporters=[exporter],
                                      name='energy-disaggregation-eval')
    if params['train']:
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # test on test data, just for CMLE online debugging's purpose
    if params['test']:
        return test(params, estimator, path, SELECT_COLUMN)


def test(hparams, estimator, test_file, SELECT_COLUMN):
    """Run trained estimator on the test set.

    Run trained estimator on the testset for debugging.

    Args:
      hparams: hyper-parameteters
      estimator: trained tf estimator
    """
    hparams['test_file'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'test2.csv')
    hparams['test_file'] = test_file
    SELECT_COLUMN = [0]
    def test_input(): return make_input_fn(
        data_file=hparams['test_file'],
        seq_len=hparams['seq_len'],
        batch_size=hparams['eval_batch_size'],
        cols=SELECT_COLUMN[:1],
        num_epochs=1)
    # load test data

    test_data = pd.read_csv(hparams['test_file'],index_col=0)

    tf.logging.info('test_data.shape={}'.format(test_data.shape))
    # make predictions
    predictions = estimator.predict(input_fn=test_input)
    print('-9-9-9-9-',predictions,'-9-9-9--9')
    preds = []
    for pred_dict in predictions:
        preds.append(pred_dict['probabilities'])
    preds = np.array(preds)
    
    tf.logging.info('preds.shape={}'.format(preds.shape))
    tf.logging.info('preds.max()={}'.format(preds.max()))
    # output metrics
    groundtruth = test_data.iloc[hparams['seq_len'] - 1:]
    pred_names = [x.replace('_on', '_pred')
                  for x in groundtruth.columns if '_on' in x]
    preds = preds.round().astype(np.uint8)
    #preds = pd.DataFrame(preds, columns=pred_names, index=groundtruth.index)
    pred_names = ['Appliance_'+str(i)+'_pred' for i in range(1,9)]
    preds = pd.DataFrame(preds,columns=pred_names)
    print('-0-0-0-',preds,'-0-0-0-0-0')
    return preds
    """
    df = pd.merge(groundtruth, preds, left_index=True, right_index=True)
    appliances_names = [x.replace('_pred', '') for x in pred_names]
    for i, app in enumerate(appliances_names):
        precision = sklearn.metrics.precision_score(
            df[app + '_on'], df[app + '_pred'])
        recall = sklearn.metrics.recall_score(
            df[app + '_on'], df[app + '_pred'])
        tf.logging.info('{0}:\tprecision={1:.2f}, recall={2:.2f}'.format(
            app, precision, recall))
    return df
    """
