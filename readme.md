再現実験をするにはハチが巣から出入りする動画が必要
IO_data/input/videosに動画を入れたらdetect_rewriteを実行する。
パスが合っているか確認する必要がある。また、Cudaのバージョンによっては動作しないので注意
cuda 11.3 Pytorch 1.11.0

requirements.txt : これをanacondaの仮想環境にインストールする必要がある。

detect_rewrite : ハチを検出して巣からの出入りをカウントする

bridge_wrapper_rewrite : ハチをカウントするためのフレーム処理をするクラスとYOLOv7でDeepSORTを使うためのクラスがある

detection_helpers : 検出器のクラス

tracking_helpers :　トラッキングするために必要なクラス
 