# `colab_utils`
This repository ontains useful scripts for adding common services to non-persistent `colaboratory` VM sessions

see: https://colab.research.google.com/notebook


## Tensorboard 
create a public `tensorboard` URL using secure introspective tunnels via `ngrok`

When training on `colaboratory` VMs it it often useful to monitor the session via 
`tensorboard`. This script helps you launches tensorboard on the `colaboratory` VM and 
uses `ngrok` to create a secure introspective tunnel to access tensorboard via public URL.

```
************************************
*     A simple working script      *
************************************

import os
import colab_utils.tboard

# set paths
ROOT = %pwd
LOG_DIR = os.path.join(ROOT, 'log')

# will install `ngrok`, if necessary
# will create `log_dir` if path does not exist
colab_utils.tboard.launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )

```

### `launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )`
launch tensorboard on `colaboratory` VM and open a tunnel for access by public URL. Automatically installs `ngrok`. if necessary.

```
tboard.launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )
```


### `install_ngrok( bin_dir=ROOT )`
install `ngrok` package, if necessary

```
tboard.install_ngrok( bin_dir=ROOT, log_dir=LOG_DIR )
```


## Google Cloud
Access Google Cloud from `colaboratory` VM and save/restore checkpoints to cloud storage

**Note:** these methods currently use ipython `magic` commands and therefore cannot be loaded from a module at this time. For now, you can copy/paste the entire script to a `colaboratory ` cell to run.


Long-running training sessions on `colaboratory` VMs are at risk of reset after 90 mins of
inactivity or shutdown after 12hrs of training. This script allows you to save/restore
checkpoints to Google Cloud Storage to avoid losing your results.

```
************************************
*     A simple working script      *
************************************

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
colab_utils.gcloud.save_to_bucket(TRAIN_LOG, bucket_name, save_events=True, force=False)


# restore a zipfile from GCS bucket to a local directory, usually in  
#     tensorboard `log_dir`
CHECKPOINTS = os.path.join(LOG_DIR, 'training-run-2')
zipfile = os.path.basename(TRAIN_LOG)   # training-run-1
colab_utils.gcloud.load_from_bucket("training-run-1.1000.zip", bucket_name, CHECKPOINTS )
```

## GCS Authorization

### `auth(project_id)`
authorize access to Google Cloud SDK from `colaboratory` VM and set default project
```
colab_utils.gcloud.gcloud_auth(project_name)
```


## Save/Restore checkpoints to Google Cloud Storage
Save and restore checkpoints and events to a zipfile in a GCS bucket


### `save_to_bucket(train_dir, bucket)`
zip the latest checkpoint files from train_dir and save to GCS bucket
```
colab_utils.gcloud.save_to_bucket(train_dir, bucket, 
                    step=None, 
                    save_events=False, 
                    force=False)
```

### `load_from_bucket(zip_filename, bucket, train_dir)`
download and unzip checkpoint files from GCS bucket, save to train_dir
```
colab_utils.gcloud.load_from_bucket(zip_filename, bucket, train_dir ):
```
