# Copyright 2018 Michael Lin. All Rights Reserved.
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
# ==============================================================================
"""Contains code for adding common services to non-persistent `colaboratory` VM sessions



  Note: these methods currently use ipython magic commands and therefore cannot be loaded 
    from a module at this time. For now, you can copy/paste the entire script to a 
    colaboratory cell to run.



  Long-running training sessions on `colaboratory` VMs are at risk of reset after 90 mins of
  inactivity or shutdown after 12hrs of training. This script allows you to save/restore
  checkpoints to Google Cloud Storage to avoid losing your results.


  ************************************
  * A simple working script *
  ************************************
  ```
  import os
  import colab_utils.gcloud

  # authorize access to Google Cloud SDK from `colaboratory` VM
  project_name = "my-project-123"
  colab_utils.gcloud.gcloud_auth(project_name)

  # set paths
  ROOT = %pwd
  LOG_DIR = os.path.join(ROOT, 'log')
  TRAIN_LOG = os.path.join(LOG_DIR, 'training-run-1')

  # save latest checkpoint as a zipfile to a GCS bucket `gs://my-checkpoints/`
  #     zipfile name = "{}.{}.zip".format() os.path.basename(TRAIN_LOG), global_step)
  #                     e.g. gs://my-checkpoints/training-run-1.1000.zip"
  bucket_name = "my-checkpoints"
  colab_utils.gcloud.push_to_bucket(TRAIN_LOG, bucket_name, save_events=True, force=False)


  # restore a zipfile from GCS bucket to a local directory, usually in  
  #     tensorboard `log_dir`
  CHECKPOINTS = os.path.join(LOG_DIR, 'training-run-2')
  zipfile = os.path.basename(TRAIN_LOG)   # training-run-1
  colab_utils.gcloud.load_from_bucket("training-run-1.1000.zip", bucket_name, CHECKPOINTS )
  ```

"""

import os
import re
import shutil
import subprocess
import tensorflow as tf

__all__ = [
  'gcloud_auth', 
  'load_from_bucket',
  'push_to_bucket',
]

def _shell(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [line for line in p.stdout.read().decode("utf-8").split("\n")]
    retval = p.wait()
    if retval==0:
        return output
    error = {'err_code': retval}  
    if p.stderr and p.stderr.read:
      error['err_msg]'] = [line for line in p.stderr.read().decode("utf-8").split("\n")]
    return error


def gcloud_auth(project_id):
  """authorize access to Google Cloud SDK from `colaboratory` VM and set default project

  Args:
    project_id: GC project

  Return:
    GCS project id
  """
  from google.colab import auth
  # authenticate user and set project
  auth.authenticate_user()
  # project_id = "my-project-123"
  get_ipython().system_raw("gcloud config set project {}".format(project_id) )
  return project_id

# tested OK
def load_from_bucket(zip_filename, bucket, train_dir):
  """download and unzip checkpoint files from GCS bucket, save to train_dir
  
  NOTE: authorize notebook before use:
    ```
    # authenticate user and set project
    from google.colab import auth
    auth.authenticate_user()
    project_id = "my-project-123"
    !gcloud config set project {project_id}
    ```

  Args:  restore from "gs://[bucket]/[zip_filename]"
    zip_filename: e.g. "my-tensorboard-run.6000.zip"
    bucket: "gs://[bucket]"
    train_dir: a diretory path to restore the checkpoint files, 
                usually TRAIN_LOG, e.g. "/my-project/log/my-tensorboard-run"
    

  Returns:
    checkpoint_name, e.g. `/my-project/log/my-tensorboard-run/model.ckpt-6000`
  
  NOTE: to restore a checkpoint, you need to write a file as follows:
  file: `/my-project/log/my-tensorboard-run/checkpoint`
    model_checkpoint_path: "/my-project/log/my-tensorboard-run/model.ckpt-6000"
    all_model_checkpoint_paths: "/my-project/log/my-tensorboard-run/model.ckpt-6000"
  """

  bucket_path = "gs://{}/".format(bucket)
  gsutil_ls = _shell("gsutil ls {}".format(bucket_path))
  if type(gsutil_ls)==dict and gsutil_ls['err_code']:
    raise ValueError("ERROR: GCS bucket not found, path={}".format(bucket_path))

  bucket_path = "gs://{}/{}".format(bucket, zip_filename)
  found = [f for f in gsutil_ls if zip_filename in f]
  if not found:
    raise ValueError( "ERROR: zip file not found in bucket, path={}".format(bucket_path))

  train_dir = os.path.abspath(train_dir)
  if not os.path.isdir(train_dir):
    raise ValueError( "invalid train_dir, path={}".format(train_dir))

  zip_filepath = os.path.join('/tmp', zip_filename)
  if not os.path.isfile( zip_filepath ):
    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    print( "downloading {} ...".format(bucket_path))
    get_ipython().system_raw( "gsutil cp {} {}".format(bucket_path, zip_filepath))
  else:
    print("WARNING: using existing zip file, path={}".format(zip_filepath))
  
  if (os.path.isdir("/tmp/ckpt")):
    shutil.rmtree("/tmp/ckpt")
  os.mkdir("/tmp/ckpt")
  print( "unzipping {} ...".format(zip_filepath))
  get_ipython().system_raw( "unzip -j {} -d /tmp/ckpt".format(zip_filepath))
  print( "installing checkpoint to {} ...".format(train_dir))
  get_ipython().system_raw( "mv /tmp/ckpt/* {}".format(train_dir))
  # example filenames:
  #   ['model.ckpt-6000.data-00000-of-00001',
  #   'model.ckpt-6000.index',
  #   'model.ckpt-6000.meta']

  # append to $train_dir/checkpoint
  # example: checkpoint_name="{train_dir}/model.ckpt-{global-step}"
  checkpoint_filename = os.path.join(train_dir, "checkpoint")
  print( "appending checkpoint to file={} ...".format(checkpoint_filename))

  global_step = re.findall(".*\.(\d+)\.zip$",zip_filename)  
  if global_step:
    checkpoint_name = os.path.join(train_dir,"model.ckpt-{}".format(global_step[0]))
  else:
    raise RuntimeError("cannot get checkpoint from zip_filename, path={}".format(zip_filename))

  if not os.path.isfile(checkpoint_filename):
    with open(checkpoint_filename, 'w') as f:
      is_checkpoint_found = False
      line_entry = 'model_checkpoint_path: "{}"'.format(checkpoint_name)
      f.write(line_entry)
  else:
    # scan checkpoint_filename for checkpoint_name
    with open(checkpoint_filename, 'r') as f:
      lines = f.readlines()
    found = [f for f in lines if os.path.basename(checkpoint_name) in f]
    is_checkpoint_found = len(found) > 0

  if not is_checkpoint_found:
    line_entry = '\nall_model_checkpoint_paths: "{}"'.format(checkpoint_name)
    # append line_entry to checkpoint_filename
    with open(checkpoint_filename, 'a') as f:
      f.write(line_entry)

  print("restored: bucket={} \n> checkpoint={}".format(bucket_path, checkpoint_name))
  return checkpoint_filename

  

# tested OK
def push_to_bucket(train_dir, bucket, step=None, save_events=True, force=False):
  """zip the latest checkpoint files from train_dir and save to GCS bucket
  
  NOTE: authorize notebook before use:
    ```
    # authenticate user and set project
    from google.colab import auth
    auth.authenticate_user()
    project_id = "my-project-123"
    !gcloud config set project {project_id}
    ```

  Args:
    train_dir: a diretory path from which to save the checkpoint files, 
                usually TRAIN_LOG, e.g. "/my-project/log/my-tensorboard-run"                
    bucket: "gs://[bucket]"
    step: global_step checkpoint number
    save_events: inclue tfevents files from Summary Ops in zip file
    force: overwrite existing bucket file

  Return:
    bucket path, e.g. "gs://[bucket]/[zip_filename]"
  """
  
  bucket_path = "gs://{}/".format(bucket)
  gsutil_ls = _shell("gsutil ls {}".format(bucket_path))
  if type(gsutil_ls)==dict and gsutil_ls['err_code']:
    raise ValueError("ERROR: GCS bucket not found, path={}".format(bucket_path))

  checkpoint_path = train_dir
  if step:
    checkpoint_pattern = 'model.ckpt-{}*'.format(step)
  else:  # get latest checkpoint
    checkpoint_pattern = os.path.basename(tf.train.latest_checkpoint(train_dir))
    
  global_step = re.findall(".*ckpt-?(\d+).*$",checkpoint_pattern)
  
  if global_step:
    zip_filename = "{}.{}.zip".format(os.path.basename(train_dir), global_step[0])
    files = ["{}/{}".format(checkpoint_path,f) for f in os.listdir(checkpoint_path) if checkpoint_pattern in f]
    # files = !ls $checkpoint_path
    print("archiving checkpoint files={}".format(files))
    filelist = " ".join(files)
    zip_filepath = os.path.join("/tmp", zip_filename)

    if save_events:
      # save events for tensorboard
      # event_path = os.path.join(train_dir,'events.out.tfevents*')
      # events = !ls $event_path
      event_pattern = 'events.out.tfevents'
      events = [f for f in os.listdir(checkpoint_path) if event_pattern in f]
      if events: 
        print("archiving event files={}".format(events))
        filelist += " " + " ".join(events)

    found = [f for f in gsutil_ls if zip_filename in f]
    if found and not force:
      raise Warning("WARNING: a zip file already exists, path={}".format(found[0]))

    print( "writing zip archive to file={} ...".format(zip_filepath))
    result = get_ipython().system_raw( "zip -D {} {}".format(zip_filepath, filelist))
    
    if not os.path.isfile(zip_filepath):
      raise RuntimeError("ERROR: zip file not created, path={}".format(zip_filepath))

    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    # result = !gsutil cp $zipfile_path $bucket_path
    print( "uploading zip archive to bucket={} ...".format(bucket_path))
    result = _shell("gsutil cp {} {}".format(zip_filepath, bucket_path))
        
    if type(result)==dict and result['err_code']:
      raise RuntimeError("ERROR: error uploading to gcloud, bucket={}".format(bucket_path))
    
    print("saved: zip={} \n> bucket={} \n> files={}".format(os.path.basename(zip_filepath), 
                                                      bucket_path, 
                                                      files))
    return bucket_path
  else:
    print("no checkpoint found, path={}".format(checkpoint_path))
    
  return


