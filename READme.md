- The `.ipynb` file utilizes `preprocessing.py` to apply preprocessing on the test data.
- As we were collaborating through Google Colab, we uploaded the necessary files to Google Drive. The following code was used to mount the Drive:
  ```
  python
  import os
  import importlib.util
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)
  ```
- This step is not necessary if running in local