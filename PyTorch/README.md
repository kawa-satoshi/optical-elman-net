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
- トークンの個数に柔軟に対応できるようになっていないため
- トークンの個数に対応したCjoinゲートの合致条件，ならびに，終了条件を設定していないため

### full_adder.pyを例に具体的に説明します．

入力位置をベタ書き（変数に直接定義）しているというのは，ソースコード中の27-47行目に相当します．
こちらをコメントアウトすることで，入力のスタート位置を変更する構造になっています．
回路情報からスタート位置を検出するようできるとありがたいです．

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

トークンの個数に柔軟に対応できるようになっていないというのは，ソースコード438-440行目のSignalのインスタンス化とinitialization関数とstart関数とdraw_signal関数に相当します．
こちらが信号トークンの数に依存したインスタンス化/引数の個数という設計になっています．
回路に依存してトークンの数が決定するため，現状では回路毎にソースコードを書き換える状況となっております．
こちらも，回路情報からトークンの数（=スタート位置の数）を入手して柔軟に対応できるとありがたいです．

```python
signal1 = Signal(x1, y1)
signal2 = Signal(x2, y2)
signal3 = Signal(x3, y3)
initialization(signal1, signal2, x1, y1, x2, y2, x3, y3)
while loop_flag:
  start(signal1, signal2, signal3)
```

トークンの個数に対応したCjoinゲートの合致条件を設定していないというのは，ソースコード363-390行目に相当します．
Cjoinゲートは2つのトークンが特定の位置に存在したときのみ”機能する”ので，どのトークンの組み合わせでゲートが”機能する”かをチェックする機構がトークンの数に依存することになります．
この組み合わせをベタに書いているので，現状では回路毎にソースコードを書き換える状況となっております．
こちらも，回路情報からトークンの数（=スタート位置の数）を入手して柔軟に対応できるとありがたいです．

```python
# Cjoin check
signal1.Cjoin_check()
signal2.Cjoin_check()
signal3.Cjoin_check()

cjoin_check_list = []
if signal1.cjoin_flag==1:
  cjoin_check_list.append(1)
if signal2.cjoin_flag==1:
  cjoin_check_list.append(2)
if signal3.cjoin_flag==1:
  cjoin_check_list.append(3)

for x in itertools.combinations(cjoin_check_list, 2): 
  if (x[0] == 1) and (x[1] == 2): 
    cjoin_execute(signal1, signal2)
  elif (x[0] == 1) and (x[1] == 3): 
    cjoin_execute(signal1, signal3)
  elif (x[0] == 2) and (x[1] == 3): 
    cjoin_execute(signal2, signal3)

# End check
signal1.End_check()
signal2.End_check()
signal3.End_check()
if (signal1.end_flag==1) and (signal2.end_flag==1) and (signal3.end_flag==1):
  print(step)
  break
```

