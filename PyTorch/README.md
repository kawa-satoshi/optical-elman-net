# Brownian Circuits Visualization

## 環境設定

Python version = 3.11.5

ライブラリーの一覧：```requirements.txt```

インストール方法：

```bash
pip install -r requirements.txt
```


## 実行

4種類のCase study回路を用意しています．

- 半加算器：half_adder
- 全加算器：full_adder
- レジスタ：register
- 3bit条件付きカウンター：3bit_conditional_counter

それぞれのディレクトリでpythonファイルを実行すればOKです．
実行例：half_adder

```bash
cd ./web_app/half_adder
python half_adder.py
```

## 入力回路の変更

入力する回路情報はそれぞれエクセルで管理しています．
マクロをONにしてファイルを開き「diagramシート」で回路図を設計します．
記載ルールは「colorシート」を参照．
設計が終わったらマクロの```diagram2schematic```を実行すると「schematicシート」が更新されます．
各ソースコードは「schematicシート」を読み込んで実行するため，編集後エクセルを保存．


## ソースコードの注意点

現状，各ソースコードは回路図に応じて作り替えています．
この理由は，下記のとおりです

- 入力位置をベタ書き（変数に直接定義）しているため
- 
- トークンの個数に柔軟に対応できるようになっていないため



```python
### 初回入力パターン
###I1: 0
x1 = 3
y1 = 8
###I1: 1
#x1 = 3
#y1 = 18

###I2: 0
x2 = 4
y2 = 1
###I2: 1
#x2 = 14
#y2 = 1

###Carry: 0
x3 = 26
y3 = 3
###Carry: 1
#x3 = 36
#y3 = 3
```

分類データセットに対して、custom活性化関数のスケールを1に固定しています。

Google's Speech Commands Datasetデータセットの入力サイズが16,000というかなり長い時系列であるため、```MAX_SEQUENCE_LENGTH```を設定しています。行った実験では最初の1,600のみを使用して、コマンドを分類しています。

### メモリー課題

対象データセット：

- メモリー（```memory```）

```run_memory.py```を実行するとfloat32とint8のモデルが```tflite_models```というフォルダに保存されます。

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

[-1, 1]一様分布に従い、ランダムにサンプリングしたデータを1つの時系列データとして使用しています。

また、```N```値を入力とし、その```K```値後の数字を予測値として設定しています。

RNNに```1```時点値から```N```時点値の数値を入力し、```N-K```時点値を直接予測します。

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

- Tensorflowのquantization手法による結果に精度が低いことを確認。
- Tensorflow-liteに変える必要があるため、RNN重みを出力する事が出来ないため、PyTorchで提供。
