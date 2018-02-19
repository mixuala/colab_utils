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
  # colab_utils.gcloud.config_project(project_name)

  # set paths
  ROOT = %pwd
  LOG_DIR = os.path.join(ROOT, 'log')
  TRAIN_LOG = os.path.join(LOG_DIR, 'training-run-1')

  

  # save latest checkpoint as a zipfile to a GCS bucket `gs://my-checkpoints/`
  #     zipfile name = "{}.{}.zip".format() os.path.basename(TRAIN_LOG), global_step)
  #                     e.g. gs://my-checkpoints/training-run-1.1000.zip"
  bucket_name = "my-checkpoints"
  colab_utils.gcloud.save_to_bucket(TRAIN_LOG, bucket_name, project_name, save_events=True, force=False)


  # restore a zipfile from GCS bucket to a local directory, usually in  
  #     tensorboard `log_dir`
  CHECKPOINTS = os.path.join(LOG_DIR, 'training-run-2')
  zipfile = os.path.basename(TRAIN_LOG)   # training-run-1
  colab_utils.gcloud.load_from_bucket("training-run-1.1000.zip", bucket_name, CHECKPOINTS )


  # mount gcs bucket to local fs using gcsfuse package 
  bucket = "my-bucket"
  lolocal_pathcaldir = colab_utils.gcloud.gcsfuse(bucket=bucket)  
  # gcsfuse(): Using mount point: /tmp/gcs-bucket/my-bucket

  !ls -l local_path
  !umount local_path


  # use `SaverWithCallback` to save tf.train.Saver() checkpoint and events as a zip archive to bucket
  #
  #
  import os, re
  import colab_utils.gcloud
  ## closure, for use with colaboratory 
  bucket = "my-bucket"
  project_name = "my-project-123"
  def save_checkpoint_to_bucket( sess, save_path, **kwargs ):
    # e.g. model_checkpoint_path = /tensorflow/log/run1/model.ckpt-14
    train_log, checkpoint = os.path.split(kwargs['checkpoint_path'])
    step = kwargs['checkpoint_step']
    bucket_path = colab_utils.gcloud.save_to_bucket(train_log, bucket, project_name, 
                                      step=step,
                                      save_events=True)
    return bucket_path

  saver = SaverWithCallback(save_checkpoint_to_bucket)
  # then call `saver.save()` as usual

  ```

"""
import os
import re
import shutil
import subprocess

from apiclient.http import MediaIoBaseDownload
from google.cloud import storage, exceptions
import tensorflow as tf

__all__ = [
  'gcloud_auth', 
  'config_project',
  'load_from_bucket',
  'load_latest_checkpoint_from_bucket',
  'save_to_bucket',
  'gcsfuse',
  'SaverWithCallback',
]

class GcsClient(object):
  """Helper class to persist project between google cloud storage calls """
  client=None

  @staticmethod
  def project(project_id=None):
    if project_id:  
      GcsClient.client = storage.Client( project=project_id )
    if GcsClient.client is None or not GcsClient.client.project:
      raise RuntimeError("Google Cloud Project is undefined. use colab_utils.gcloud.config_project(project_id)")
    return GcsClient.client.project  

def __shell__(cmd, split=True):
  # get_ipython().system_raw(cmd)
  result = get_ipython().getoutput(cmd, split=split)
  if result and not split:
    result = result.strip('\n')
  return result

def config_project(project_id=None):
  """called by gcloud_auth()
  """
  return GcsClient.project(project_id)



def gsutil_ls(bucket_name, filter=None, project_id=None):
  """
  list "files" in gcs bucket
  test for NotFound using 

  result = gsutil_ls(bucket_name)
  notFound = "BucketNotFoundException" in result

  Args:
    bucket_name
    filter: filter file list for string, no wildcards
    project_id: gcs project_id

  Return:
    SList() of file names or ["BucketNotFoundException", "GCS bucket not found, path={}"]
  """
  if project_id is None:
    client = GcsClient.client
  else:
    client = storage.Client( project=project_id )


  try:
    # client = storage.Client( project=project_id )
    bucket_path = "gs://{}/".format(bucket_name)
    bucket = client.get_bucket(bucket_name)
    files = ["{}{}".format(bucket_path,f.name) for f in bucket.list_blobs() ]
    if filter:
      files = [f for f in files if filter in f]
    # print(files)
    return files

  except exceptions.NotFound:
    return ["BucketNotFoundException", "GCS bucket not found, path={}".format(bucket_path)]
  except Exception as e:
    raise e

def gsutil_mb(bucket_name, gcs_class="regional", gcs_location="asia-east1", project_id=None):
  """create a bucket in the GCS project

  same as `!gsutil mb -p {project_id} -c {gcd_class} -l {gcs_location} {bucket_name}` 
  
  Args:
    bucket_name: bucket name
    gcs_class: storage class if creating bucket [standard|regional|etc]
    gcs_location: storage location, if creating bucket [asia-east1|etc]

  Return:
    GCS bucket path, e.g. gs://{bucket}
  """

  _cmd = {
    "make_bucket"             :  "gsutil mb -p {} -c {} -l {}    {}",
  }


  if project_id is None:
    client = GcsClient.client
    project_id = client.project
  else:
    client = storage.Client( project=project_id )

  ###
  ### TODO: how to set class, location using client.create_bucket()??
  ###
  # try:
  #   # client = storage.Client( project=project_id )
  #   bucket_path = client.create_bucket(bucket_name)
  #   # TODO: check above line
  # except exceptions.Conflict:
  #   raise ValueError("ERROR: GCS bucket exists, path={}".format(bucket_path))
  # except Exception as e:
  #   print(e)

  BUCKET = bucket_name
  BUCKET_PATH = "gs://{}".format(BUCKET)
  result = gsutil_ls(BUCKET)
  # result = __shell__("gsutil ls {}".format(BUCKET_PATH, split=False))
  if "BucketNotFoundException" in result: 
    print("making bucket={}".format(BUCKET_PATH))
    # TODO: use gcs python API (above)
    cmd = _cmd["make_bucket"].format( project_id, gcs_class, gcs_location, BUCKET_PATH)
    # print(cmd)
    result = __shell__(  cmd, split=True )
    print("\n".join(result))
    if len(result) == 1 and "Creating gs://" in result[0]:
      return BUCKET_PATH
    raise RuntimeError("ERROR: problem creating bucket, bucket={}".format(BUCKET_PATH))
  else:
    raise ValueError("ERROR: gcs bucket exists, bucket={}".format(BUCKET_PATH))


def gcs_download(gcs_path, local_path, project_id=None, force=False):
  bucket_path, filename = os.path.split(gcs_path)
  bucket_name = os.path.basename(bucket_path)
  if os.path.isfile(local_path) and not force:
    raise Warning("WARNING: local file already exists, use force=True. path={}".format(local_path))
  
  if project_id is None:
    client = GcsClient.client
  else:
    client = storage.Client( project=project_id )

  try:
    # client = storage.Client( project=project_id )
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(filename, bucket)
    print("downloading file={} ...".format(gcs_path))
    blob.download_to_filename(local_path)
    return local_path

  except exceptions.NotFound:
    raise ValueError("BucketNotFoundException: GCS bucket not found, path={}".format(bucket_path))
  except Exception as e:
    print(e)




def gcs_upload(local_path, gcs_path, project_id=None, force=False):
  bucket_path, filename = os.path.split(gcs_path)
  bucket_name = os.path.basename(bucket_path)
  
  if project_id is None:
    client = GcsClient.client
  else:
    client = storage.Client( project=project_id )

  try:
    result = gsutil_ls(bucket_name, filter=filename, project_id=project_id)
    # result = __shell__("gsutil ls {}".format(BUCKET_PATH, split=False))
    if "BucketNotFoundException" in result: 
      raise ValueError( "ERROR: bucket not found, path={}".format(bucket_name))
    if result and not force:
      raise Warning("WARNING: gcs file already exists, use force=True. bucket={}".format(bucket_name))

    # client = storage.Client( project=project_id )
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(filename, bucket)
    print("uploading file={} ...".format(gcs_path))
    blob.upload_from_filename(local_path)
    return gcs_path

  except exceptions.NotFound:
    raise ValueError("BucketNotFoundException: GCS bucket not found, path={}".format(bucket_path))
  except Exception as e:
    print(e)


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
  config_project(project_id)  # set for google.cloud.storage
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

  Args:  
    zip_filename: e.g. "my-tensorboard-run.6000.zip"
    bucket: restore from "gs://[bucket]/[zip_filename]"
    train_dir: a diretory path to restore the checkpoint files, 
                usually TRAIN_LOG, e.g. "/my-project/log/my-tensorboard-run"
    

  Returns:
    checkpoint_name, e.g. `/my-project/log/my-tensorboard-run/model.ckpt-6000`
  
  NOTE: to restore a checkpoint, you need to write a file as follows:
  file: `/my-project/log/my-tensorboard-run/checkpoint`
    model_checkpoint_path: "/my-project/log/my-tensorboard-run/model.ckpt-6000"
    all_model_checkpoint_paths: "/my-project/log/my-tensorboard-run/model.ckpt-6000"
  """

  # bucket_path = "gs://{}/".format(bucket)
  # found = _shell("gsutil ls {}".format(bucket_path))
  bucket_path = "gs://{}/{}".format(bucket, zip_filename)

  found = gsutil_ls(bucket, filter=zip_filename)
  if "BucketNotFoundException" in found: 
    raise ValueError( "ERROR: bucket not found, path={}".format(bucket))
  if not found:
    raise ValueError( "ERROR: zip file not found in bucket, path={}".format(bucket_path))

  train_dir = os.path.abspath(train_dir)
  if not os.path.isdir(train_dir):
    raise ValueError( "invalid train_dir, path={}".format(train_dir))

  zip_filepath = os.path.join('/tmp', zip_filename)
  if not os.path.isfile( zip_filepath ):
    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    print( "downloading {} ...".format(bucket_path))
    # get_ipython().system_raw( "gsutil cp {} {}".format(bucket_path, zip_filepath))
    result = gcs_download(bucket_path, zip_filepath)
  else:
    print("WARNING: using existing zip file, path={}".format(zip_filepath))
  
  if (os.path.isdir("/tmp/ckpt")):
    shutil.rmtree("/tmp/ckpt")
  os.mkdir("/tmp/ckpt")
  print( "unzipping {} ...".format(zip_filepath))
  # unzip -j ignore directories, -d target dir, tar -xvfz {archive.tar.gz} --overwrite --directory {target}
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



def load_latest_checkpoint_from_bucket(tensorboard_run, bucket, train_dir):
  """find latest zipped 'checkpoint' in bucket and download
    similar to tf.train.latest_checkpoint()

  Args:
    tensorboard_run: filter for zip files from the same run 
        e.g.  "y-tensorboard-run" for  "my-tensorboard-run.6000.zip"
    bucket: "gs://[bucket]"
    train_dir: a diretory path to restore the checkpoint files, 
                usually TRAIN_LOG, e.g. "/my-project/log/my-tensorboard-run"

  Return:
    checkpoint_name, e.g. `/my-project/log/my-tensorboard-run/model.ckpt-6000`
  """
  import numpy as np
  checkpoints = gsutil_ls(bucket, filter=tensorboard_run)
  if "BucketNotFoundException" in checkpoints: 
    raise ValueError( "ERROR: bucket not found, path={}".format(bucket))
  if not checkpoints:
    raise ValueError("Checkpoint not found, tensorboard_run={}".format(tensorboard_run))
  steps = [re.findall(".*\.(\d+)\.zip$", f)[0] for f in checkpoints ]
  if not steps:
    raise ValueError("Checkpoint not found, tensorboard_run={}".format(tensorboard_run))
  latest_step = np.max(np.asarray(steps).astype(int))
  if not latest_step:
    raise ValueError("Checkpoint not found, tensorboard_run={}".format(tensorboard_run))
  latest_checkpoint = [f for f in checkpoints if latest_step.astype(str) in f ]
  print("latest checkpoint found, checkpoint={}".format(latest_checkpoint[0]))
  zip_filename = os.path.basename(latest_checkpoint[0])
  return load_from_bucket(zip_filename, bucket, train_dir)

    

# tested OK
def save_to_bucket(train_dir, bucket, project_id, basename=None, step=None, save_events=True, force=False):
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
    project_id: GCS project_id 
      # Note: pass explicitly because GcsClient.client doesn't seem to be working. timeout? 
    basename: basename for the zip archive, e.g. filename="{basename}.{global_step}.zip"
      default to os.path.basename(train_dir), or the tensorboard log dir
    step: global_step checkpoint number, if None, then use result from `tf.train.latest_checkpoint()`
    save_events: include tfevents files from Summary Ops in zip file
    force: overwrite existing bucket file

  Return:
    bucket path, e.g. "gs://[bucket]/[zip_filename]"
  """

  checkpoint_path = train_dir
  if step:
    checkpoint_pattern = 'model.ckpt-{}'.format(step)
  else:  # get latest checkpoint
    checkpoint = tf.train.latest_checkpoint(train_dir)
    if checkpoint==None:
      raise RuntimeError("Error: tf.train.latest_checkpoint()")
    checkpoint_pattern = os.path.basename(checkpoint)
    
  global_step = re.findall(".*ckpt-?(\d+).*$",checkpoint_pattern)
  
  if global_step:
    if basename is None:
      basename = os.path.basename(train_dir)
    zip_filename = "{}.{}.zip".format(basename, global_step[0])
    zip_filepath = os.path.join("/tmp", zip_filename)

    # check if gcs file already exists
    # bucket_path = "gs://{}/".format(bucket)
    # bucket_files = _shell("gsutil ls {}".format(bucket_path))
    found = gsutil_ls(bucket, filter=zip_filename, project_id=project_id)
    if "BucketNotFoundException" in found: 
      raise ValueError( "ERROR: bucket not found, path={}".format(bucket))
    if found and not force:
      raise RuntimeError("WARNING: a zip file already exists, path={}. use force=True to overwrite".format(found[0]))
      
    files = ["{}/{}".format(checkpoint_path,f) for f in os.listdir(checkpoint_path) if checkpoint_pattern in f]
    # files = !ls $checkpoint_path
    print("archiving checkpoint files={}".format(files))
    filelist = files
  

    if save_events:
      # save events for tensorboard
      # event_path = os.path.join(train_dir,'events.out.tfevents*')
      # events = !ls $event_path
      event_pattern = 'events.out.tfevents'
      events = ["{}/{}".format(checkpoint_path,f) for f in os.listdir(checkpoint_path) if event_pattern in f]
      if events: 
        print("archiving event files={}".format(events))
        filelist = files + events


    print( "writing zip archive to, file={}, count={} ...".format(len(filelist), zip_filepath))
    # zip -D no dirs, tar -czvf {zip_filepath.tar.gz} -C {checkpoint_path} [f for f in os.listdir(...)]
    result = get_ipython().system_raw( "zip -D {} {}".format(zip_filepath, " ".join(filelist)))
    
    if not os.path.isfile(zip_filepath):
      raise RuntimeError("ERROR: zip file not created, path={}".format(zip_filepath))

    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    print( "uploading zip archive to bucket={} ...".format(bucket_path))
    # result = _shell("gsutil cp {} {}".format(zip_filepath, bucket_path))
    result = gcs_upload(zip_filepath, bucket_path, project_id=project_id)
        
    if type(result)==dict and result['err_code']:
      raise RuntimeError("ERROR: error uploading to gcloud, bucket={}".format(bucket_path))
    
    print("saved: zip={} \n> bucket={} \n> files={}".format(os.path.basename(zip_filepath), 
                                                      bucket_path, 
                                                      files))
    return bucket_path
  else:
    print("no checkpoint found, path={}".format(checkpoint_path))
    
  return


  



def gcsfuse(bucket=None, gcs_class="regional", gcs_location="asia-east1", project_id=None):
  """install `gcsfuse` as necessary and mount GCS bucket to local fs in `/tmp/gcs-bucket/[bucket]`

  NOTE: not sure how to use client.create_bucket() with class and location defaults ?
  see: https://stackoverflow.com/questions/48728491/google-cloud-storage-api-how-do-you-call-create-bucket-with-storage-class-a

  Args:
    bucket: bucket name
    gcs_class: storage class if creating bucket [standard|regional|etc]
    gcs_location: storage location, if creating bucket [asia-east1|etc]

  Return:
    path to local fs dir (fused to bucket), FUSED_BUCKET_PATH
  """

  ### cmd strings passed to shell
  _cmd = {
    "install_lsb_release" : "apt-get -y install lsb-release",
    "get_lsb_release"     : "lsb_release -c -s",
    "install_gcsfuse"     :[ 
                            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -",
                            "apt-get update",
                            "apt-get -y install gcsfuse",
                           ],
    "gcsfuse"             :  "gcsfuse {} {}",
  }

  found = os.path.isfile("/usr/bin/gcsfuse")
  if not found:
    ###
    ### install gcsfuse
    ###
    print("installing gcsfuse...")
    __shell__( _cmd["install_lsb_release"] )
    GCSFUSE_REPO = "gcsfuse-{}".format( __shell__(_cmd["get_lsb_release"], split=False))
    # add package to distro
    line_entry = "deb http://packages.cloud.google.com/apt {} main" .format(GCSFUSE_REPO)
    # append line_entry to apt sources file
    filepath = "/etc/apt/sources.list.d/gcsfuse.list"
    with open(filepath, 'a') as f: f.write(line_entry)
    
    for cmd in _cmd["install_gcsfuse"]:
      __shell__( cmd )

    found = os.path.isfile("/usr/bin/gcsfuse")
    if not found:
      raise RuntimeError("ERROR: problem installing gcsfuse")
    print("gcsfuse installation complete:  /usr/bin/gcsfuse")

  if project_id is None:
    project_id = config_project()

  ###
  ### get valid google cloud BUCKET_PATH, create if necessary 
  ###
  if bucket:
    BUCKET = bucket
  else:
    import time
    BUCKET = "gcsfuse-{}".format(int(time.time()))
    
  ### get bucket, create if necessary
  BUCKET_PATH = "gs://{}".format(BUCKET)
  result = gsutil_ls(BUCKET)
  if "BucketNotFoundException" in result:
    result = gsutil_mb(BUCKET, project_id=project_id)
  print("gsutil ls {}: {} ".format(BUCKET_PATH, result))

  ### fuse bucket to local fs
  FUSED_BUCKET_PATH = "/tmp/gcs-bucket/{}".format(BUCKET)
  if not tf.gfile.Exists(FUSED_BUCKET_PATH):  tf.gfile.MakeDirs(FUSED_BUCKET_PATH)
  # cmd = _cmd["gcsfuse"].format(BUCKET, FUSED_BUCKET_PATH)
  # print(cmd)
  result = __shell__(  _cmd["gcsfuse"].format(BUCKET, FUSED_BUCKET_PATH)  )
  print("gcsfuse():\n", "\n".join(result))
  if result.pop()=='File system has been successfully mounted.':
    return FUSED_BUCKET_PATH
  raise RuntimeError("ERROR: problem mounting gcs, bucket={}\n{}".format(BUCKET, "\n".join(result)))








class SaverWithCallback(tf.train.Saver):
  """override tf.train.Saver to call `callback_op` after tf.train.Saver.save()

    pass `callback_op(sess, save_path, **kwargs)` as the first arg to the constructor, or 
    call self.set_callback(callback_op)

    example:
      ```
      def after_save(sess, save_path, **kwargs):
        step = kwargs['checkpoint_step']
        path = kwargs['checkpoint_path']
        print("SaverWithCallback.save() returned with checkpoint, step={},  path={}".format(step, path))

      saver = SaverWithCallback(after_save)
      ```  
  """
  _callback_op = None
  def __init__(self, callback_op, **kwargs ):
      self._callback_op = callback_op
      super().__init__(**kwargs)

  def set_callback(self, callback_op):
    self._callback_op = callback_op
      
  def save(self, sess, save_path, **kwargs):
      """override tf.train.Saver, call callback_op() after tf.train.Saver.save()
      see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/saver.py
      """
      model_checkpoint_path = super().save(sess, save_path, **kwargs)
      if self._callback_op is not None:
          ## call on a new thread?
          try:
            train_log, checkpoint = os.path.split(model_checkpoint_path)
            found = re.findall(".*\.ckpt-(\d+)$",checkpoint)
            step = found[0] if found else None
            kwargs = dict(kwargs, checkpoint_path=model_checkpoint_path, checkpoint_step=step)
            self._callback_op(sess, save_path, **kwargs)
          except Exception as e:
            print("WARNING: SaverWithCallback() callback exception, err={}".format(e))
      return model_checkpoint_path
    