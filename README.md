# RNNの実験モデル

データセット：S&P500

## 環境設定

Python version = 3.10.12

ライブラリーの一覧：```requirements.txt```ですが、
管理のためにこのプロジェクトでPoetryというライブラリーを使用しています。

インストール方法：<https://python-poetry.org/docs/>

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

出力されたコメントを確認して、```poetry```を環境変数に追加します。

```bash
poetry install
poetry shell
```

## 実行

実行スクリプトを3つに分けています：「回帰」、「分類」と「メモリー」。

- 回帰課題として旅客数予測とS&P500予測を実装させていただいております。
- 分類課題としてGoogle's Speech Commands DatasetとIrisを実装させていただいております。
  - 最後の層の後にSoftmax関数を追加しております。
- 最後はメモリー課題を別スクリプトとして提供させていただきます。

### 回帰データセット

対象データセット：

- S&P500予測（```SP500```）
- 旅客数予測（```airline-passengers```）

```run_regression.py```を実行しますとfloat32とint8のモデルが```tflite_models```というフォルダに保存されます。

実行ファイル：

```bash
python run_regression.py
```

設定パラメータは以下：

```python
DATASET_NAME = "SP500"
# DATASET_NAME = "airline-passengers"

EPOCHS = 100
ACTIVATION = "custom"
custom_max_value = 255.0
DATA_SIGMA = 255.0
HIDDEN_SIZE = 10
SEQUENCE_LENGTH = 20
```

### 分類データセット

対象データセット：

- Google's Speech Commands Dataset（```speech_commands```）
- Iris（```iris```）

```run_classification.py```を実行しますとfloat32とint8のモデルが```tflite_models```というフォルダに保存されます。

実行ファイル：

```bash
python run_classification.py
```

設定パラメータは以下：

```python
DATASET_NAME = "speech_commands"
# DATASET_NAME = "iris"

EPOCHS = 100
ACTIVATION = "custom"
HIDDEN_SIZE = 10
MAX_SEQUENCE_LENGTH = 1600
```

分類データセットに対して、custom活性化関数のスケールを1に固定しています。

Google's Speech Commands Datasetデータセットの入力サイズが16,000というかなり長い時系列であるため、```MAX_SEQUENCE_LENGTH```を設定させていただいております。行った実験では最初の1,600のみを使用して、コマンドを分類しています。

### メモリー課題

対象データセット：

- メモリー（```memory```）

```run_memory.py```を実行しますとfloat32とint8のモデルが```tflite_models```というフォルダに保存されます。

実行ファイル：

```bash
python run_memory.py
```

設定パラメータは以下：

```python
DATASET_NAME = "memory"

EPOCHS = 100
ACTIVATION = "tanh"
custom_max_value = 1.0
HIDDEN_SIZE = 10

MEMORY_SIGNAL_LENGTH = 1024
MEMORY_N = 100
MEMORY_K = 1
```

[-1, 1]一様分布に従い、ランダムにサンプリングしたデータを1つの時系列データとして使用しております。

また、```N```値を入力とし、その```K```値後の数字を予測値として設定しています。

RNNに```1```時点値から```N```時点値の数値を入力し、```N+1```から```N+K```までではなく、```N+K```時点値を直接予測します。

また、1つのランダム生成された時系列データを覚えることが課題であるため、学習データとバリデーションデータを分けないことが適切だと判断しました。

メモリーデータセットに対して、[-1, 1]一様分布を使用のため、custom活性化関数のスケールは1でのご利用を予想しています。

## 実装済み実験について

使用可能のデータセット

- S&P500予測（SP500）
- 旅客数予測（airline-passengers）
- Google's Speech Commands Dataset（speech_commands）
- Iris（iris）
- メモリー（memory）

使用可能のactivation function：

- relu
- tanh
- custom:（0-<custom_max_value>）でのスケールsqrt

学習終了後、モデルのint8化を行いますが、ニューロン数と層数が少ないため、結果の劣化が非常に大きいことを確認しました。

## Int8モデルについて

- Tensorflowのquantization手法による結果に精度が低いことを確認しています。
- Tensorflow-liteに変える必要があるため、RNN重みを出力する事が出来ないため、PyTorchで提供させてください。
