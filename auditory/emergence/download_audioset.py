# # Install audioset-download
# pip install audioset-download

# 解决报错：joblib.externals.loky.process_executor.TerminatedWorkerError: A worker...
# pip install joblib==1.2.0
from audioset_download import Downloader
d = Downloader(root_path='test', labels=None, n_jobs=2, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'vorbis')
