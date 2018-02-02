# colab_utils
assorted utils for use with `Colaboratory`

see: https://colab.research.google.com/notebook

## GCS Authorization
### `google_cloud_auth(project_id)`
authenticate Google user for access from within `Colaboratory` notebook and set default project

    ```
    import colab_utils
    colab_utils.google_cloud_auth("my-project-123")
    ```


## Persistence between VM sessions
> Note: authorize access to Google account before use

### `save_to_bucket(train_dir, bucket)`

```
save_to_bucket(train_dir, bucket, 
                    step=None, 
                    save_events=False, 
                    force=False)
```

save checkpoints and events to a zipfile in a GCS bucket
  ```
  # example:
  train_dir = "/my-project/log/my-tensorboard-run"
  # `gs://my-project/my-bucket`
  bucket = "my-bucket"
  
  colab_utils.save_to_bucket(train_dir, bucket, step=None, save_events=True, force=False)
  ```

### `load_from_bucket(zip_filename, bucket, train_dir)`
load a saved checkpoint from a GCS bucket to a training directory for restore
  ```
  # example
  zip_filename = "my-tensorboard-run.6000.zip"
  # `gs://my-project/my-bucket`
  bucket = "my-bucket"
  train_dir = "/my-project/log/my-tensorboard-run"
  
  colab_utils.load_from_bucket(zip_filename, bucket, train_dir ):
  ```