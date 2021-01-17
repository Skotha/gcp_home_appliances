import os
import tensorflow as tf
from m1 import run_experiment
SEQ_LEN = 5
SELECT_COLUMN = list(range(8, 17))
SELECT_COLUMN = [19, ] + list(range(21, 29))
feat_col_names = ['ActivePower_{}'.format(i + 1)
                  for i in range(SEQ_LEN)]
params = {'feat_cols': feat_col_names,
          'seq_len': SEQ_LEN,
          'lstm_size': 131,
          'batch_size': 64,
          'num_appliances': len(SELECT_COLUMN) - 1,
          'num_layers': 4,
          'learning_rate': 8.1729e-5,
          'dropout_rate': 0.3204,
          'use_keras': True,
          'keras': True,
          'train_file': 'gs://gcp_blog/e2e_demo/train.csv',
          # 'train_file': 'gs://gcp-project-290817/data/train_power_data_home_1.csv',
          'num_epochs': 40,
          'train_batch_size': 64,
          'eval_batch_size': 64,
          'eval_file': 'gs://gcp_blog/e2e_demo/eval.csv',
          'test_file': 'gs://gcp_blog/e2e_demo/test.csv',
          'job_dir': 'model',
          # 'job_dir' : '/tmp',
          'model_name': 'home_appliances_predict',
          'verbosity': 'INFO',
          'train_steps': 1e5,
          'eval_steps': 1e3,
          'filter_prob': 0.6827,
          'test': True,
          'train': False,
          'keras': True,
          }
# Set python level verbosity
tf.logging.set_verbosity(params['verbosity'])
# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
    tf.logging.__dict__[params['verbosity']] / 10)
for k, v in params.items():
    tf.logging.info('{}: {}'.format(k, v))


def go(path):
    return run_experiment(params, path, SELECT_COLUMN)
