"""

this script is imcomplete.

in the process of converting ipython `magic` commands to native python to allow for 
module import.

"""


import os
import requests
import shutil
import subprocess
import tensorflow as tf
from google.colab import auth

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


def google_cloud_auth(project_id):
  """authorize Google user and set default project

  Args:
    project_id: GC project
  """
  # authenticate user and set project 
  auth.authenticate_user()
  # project_id = "my-project-123"
  get_ipython().system_raw("gcloud config set project {}".format(project_id) )
  return project_id



# ready to test
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
    zip_filename = "{}.{}.zip".format(os.path.basename(TRAIN_LOG), global_step[0])
    files = [f for f in os.listdir(checkpoint_path) if checkpoint_pattern in f]
    # files = !ls $checkpoint_path
    print("archiving checkpoint files={}".format(files))
    filelist = " ".join(files)
    zipfile_path = os.path.join("/tmp", zip_filename)

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
      raise ValueError("WARNING: a zip file already exists, path={}".format(found[0]))

    # !zip  $zipfile_path $filelist
    print( "writing zip archive to file={} ...".format(zip_filepath))
    get_ipython().system_raw( "zip {} {}".format(zip_filepath, filelist))
    bucket_path = "gs://{}/{}".format(bucket, zip_filename)
    # result = !gsutil cp $zipfile_path $bucket_path
    result = = _shell("gsutil cp {} {}".format(zip_filepath, bucket_path))
    print("saved: zip={} \n> bucket={} \n> files={}".format(zipfile_path, bucket_path, files))
    return bucket_path
  else:
    print("no checkpoint found, path={}".format(checkpoint_path))




    


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

  #  append to train_dir/checkpoint
  checkpoint_filename = os.path.join(train_dir, "checkpoint")
  print( "appending checkpoint to file={} ...".format(checkpoint_filename))
  checkpoint_name = [f for f in os.listdir(train_dir) if ".meta" in f]
  checkpoint_name = checkpoint_name[0][:-5]   # pop() and slice ".meta"
  checkpoint_name = os.path.join(train_dir,os.path.basename(checkpoint_name))

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
def install_ngrok(bin_dir="/tmp"):
  ROOT = bin_dir
  CWD = os.getcwd()
  is_grok_avail = os.path.isfile(os.path.join(ROOT,'ngrok'))
  if is_grok_avail:
    print("ngrok installed")
  else:
    import platform
    plat = platform.platform() # 'Linux-4.4.64+-x86_64-with-Ubuntu-17.10-artful'
    if 'x86_64' in plat:
      
      os.chdir('/tmp')
      print("calling wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip ..." )
      get_ipython().system_raw( "wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip" )
      print("calling unzip ngrok-stable-linux-amd64.zip ...")
      get_ipython().system_raw( "unzip ngrok-stable-linux-amd64.zip" )
      os.rename("ngrok", "{}/ngrok".format(ROOT))
      os.remove("ngrok-stable-linux-amd64.zip")
      is_grok_avail = True
      os.chdir(CWD)
      if ([f for f in os.listdir() if "ngrok" in f]):
        print("ngrok installed at: {}/ngrok".format(ROOT))
      else:
        raise ValueError( "ERROR: ngrok not found, path=".format(CWD) )
    else:
      raise ValueError( "ERROR, ngrok install not configured for this platform, platform={}".format(plat))

    
# tested OK
def launch_tensorboard(bin_dir="/tmp", log_dir="/tmp"):
  """returns a public tensorboard url based on the ngrok package

  see: https://stackoverflow.com/questions/47818822/can-i-use-tensorboard-with-google-colab

  Args:
    bin_dir: full path to directory for installing ngrok package

  Return:
    public url for tensorboard

  """
  install_ngrok(bin_dir)
    
  if not tf.gfile.Exists(LOG_DIR):  tf.gfile.MakeDirs(LOG_DIR)
  
  # check status of tensorboard and ngrok
  ps = _shell("ps -ax")
  is_tensorboard_running = len([f for f in ps if "tensorboard" in f ]) > 0
  is_ngrok_running = len([f for f in ps if "ngrok" in f ]) > 0
  print("status: tensorboard={}, ngrok={}".format(is_tensorboard_running, is_ngrok_running))

  if not is_tensorboard_running:
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
        .format(log_dir)
    )
    is_tensorboard_running = True
    
  if not is_ngrok_running:  
    #    grok should be installed in /tmp/ngrok
    get_ipython().system_raw('{}/ngrok http 6006 &'.format(bin_dir))
    is_ngrok_running = True

  # get tensorboard url
  # BUG: getting connection refused for HTTPConnectionPool(host='localhost', port=4040)
  #     on first run, retry works
  import time
  time.sleep(3)
  retval = requests.get('http://localhost:4040/api/tunnels')
  tensorboard_url = retval.json()['tunnels'][0]['public_url'].strip()
  print("tensorboard url=", tensorboard_url)
  return