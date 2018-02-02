import os
import tensorflow as tf
from google.colab import auth

def google_cloud_auth(project_id):
  """authorize Google user and set default project

  Args:
    project_id: GC project
  """
  # authenticate user and set project 
  auth.authenticate_user()
  # project_id = "my-project-123"
  !gcloud config set project {project_id}
  return project_id


def save_to_bucket(train_dir, bucket, step=None, save_events=False, force=False):
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
  bucket_path = "gs://{}".format(bucket)
  gsutil_ls = !gsutil ls $bucket_path
  if "BucketNotFoundException" in gsutil_ls[0]:
    raise ValueError("ERROR: GCS bucket not found, path=".format(bucket_path))


  if step:
    checkpoint_path = os.path.join(train_dir,'model.ckpt-{}*'.format(step))
  else:
    checkpoint_path = tf.train.latest_checkpoint(train_dir) + "*"
    
  global_step = re.findall(".*ckpt-?(\d+).*$",checkpoint_path)
  
  if global_step:
    zip_filename = "{}.{}.zip".format(os.path.basename(TRAIN_LOG), global_step[0])
    files = !ls $checkpoint_path
    filelist = " ".join(files)
    zipfile_path = os.path.join("/tmp", zip_filename)

    if save_events:
      # save events for tensorboard
      event_path = os.path.join(train_dir,'events.out.tfevents*')
      events = !ls $event_path
      if events: 
        filelist += " " + " ".join(events)

    found = [f for f in gsutil_ls if zip_filename in f]
    if found and not force:
      raise ValueError("WARNING: a zip file already exists, path={}".format(found[0]))

    !zip  $zipfile_path $filelist
    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    result = !gsutil cp $zipfile_path $bucket_path
    print("saved: zip={} \n> bucket={} \n> files={}".format(zipfile_path, bucket_path, files))
    return bucket_path
  else:
    print("no checkpoint found, path={}".format(checkpoint_path))


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

  bucket_path = "gs://{}".format(bucket)
  gsutil_ls = !gsutil ls $bucket_path
  if "BucketNotFoundException" in gsutil_ls[0]:
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
    !gsutil cp $bucket_path $zip_filepath
  else:
    print("WARNING: using existing zip file, path={}".format(zip_filepath))

  !rm -rf /tmp/ckpt
  !mkdir /tmp/ckpt
  !unzip -j $zip_filepath -d /tmp/ckpt
  !cp /tmp/ckpt/* $train_dir
  # example filenames:
  #   ['model.ckpt-6000.data-00000-of-00001',
  #   'model.ckpt-6000.index',
  #   'model.ckpt-6000.meta']

  #  append to train_dir/checkpoint
  checkpoint_filename = os.path.join(train_dir, "checkpoint")
  checkpoint_name = !ls /tmp/ckpt/*.meta
  checkpoint_name = checkpoint_name[0][:-5]   # pop() and slice ".meta"
  checkpoint_name = os.path.join(train_dir,os.path.basename(checkpoint_name))

  if not os.path.isfile(checkpoint_filename):
    with open(checkpoint_filename, 'w') as f:
      is_checkpoint_found = False
      line_entry = 'model_checkpoint_path: "{}"'.format(checkpoint_name)
      f.write(line_entry)
  else:
    # scan checkpoint_filename for checkpoint_name
    lines = !cat $checkpoint_filename
    found = [f for f in lines if os.path.basename(checkpoint_name) in f]
    is_checkpoint_found = len(found) > 0

  if not is_checkpoint_found:
    line_entry = '\nall_model_checkpoint_paths: "{}"'.format(checkpoint_name)
    # append line_entry to checkpoint_filename
    with open(checkpoint_filename, 'a') as f:
      f.write(line_entry)

  print("restored: bucket={} \n> checkpoint={}".format(bucket_path, checkpoint_name))
  return checkpoint_filename

def install_ngrok(bin_dir="/tmp"):
  ROOT = bin_dir
  CWD = %pwd
  import os
  is_grok_avail = os.path.isfile(os.path.join(ROOT,'ngrok'))
  if is_grok_avail:
    print("grok installed")
  else:
    import platform
    plat = platform.platform() # 'Linux-4.4.64+-x86_64-with-Ubuntu-17.10-artful'
    if `Linux` in plat and `x86_64` in plat:
      %cd $dir
      ! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
      ! unzip ngrok-stable-linux-amd64.zip
      %mv ngrok $ROOT
      %rm ngrok-stable-linux-amd64.zip
      is_grok_avail = True
      %cd $CWD
    else:
      raise ValueError( "ERROR, ngrok install not configured for this platform, platform={}".format(plat))

def launch_tensorboard(bin_dir="/tmp"):
  """returns a public tensorboard url based on the ngrok package

  see: https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab

  Args:
    bin_dir: full path to directory for installing ngrok package

  Return:
    public url for tensorboard

  """
  install_ngrok(bin_dir)
  
  # check status of tensorboard and ngrok
  ps = !ps -ax
  is_tensorboard_running = len([f for f in ps if "tensorboard" in f ]) > 0
  is_ngrok_running = len([f for f in ps if "ngrok" in f ]) > 0
  print("status: tensorboard={}, ngrok={}".format(is_tensorboard_running, is_ngrok_running))

  if not is_tensorboard_running:
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
        .format(LOG_DIR)
    )
    is_tensorboard_running = True
    
  if not is_ngrok_running:  
    #    grok should be installed in /tmp/ngrok
    get_ipython().system_raw('{}/ngrok http 6006 &'.format(bin_dir))
    is_ngrok_running = True

  # get tensorboard url
  tensorboard_url = !curl -s http://localhost:4040/api/tunnels | python3 -c \
      "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"  
  tensorboard_url = tensorboard_url[0].strip()  
  print("tensorboard url=", tensorboard_url)
  return tensorboard_url

