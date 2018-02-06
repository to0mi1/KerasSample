Keras を使った DNN 画像分類
==

Cifer-10 を参考に Keras で実装。

モデルの生成
--

- create_model_type1.py
  - 今までの cifar-10 と同じモデル
- create_model_type2.py
  - これから検証する少し小さくしたモデル

### create_model_type2.py について

#### 変更点

- 「畳み込み層 ⇒ 畳み込み層 ⇒ プーリング層」としていたのを
「畳み込み層 ⇒ プーリング層」とし畳み込み層を一つ減じた
- 全結合層の次元数を 512 から 64 へ大幅に減少


実行について
--

create_model_type[12].py は引数を与えずに実行出来ます。

### 出力ファイル

`./model/yyyymmddHHMMSS/` ディレクトリに以下の 2 つのファイルが出力される。

- `*_model.png`
  - 作成したモデルのプロット
  - 見やすくするために出力するもので プログラムでは使用しない
- `*_model.json`
  - 作成したモデルのオブジェクト
  - トレーニング・予測の時に使用する

特に json ファイルは保管する必要があります。

json ファイルを参照すると、各層の設定値を参照することが出来ます。
何か Excel 等に分かり易くまとめておくのが Best ですがそこまでは自動化出来ていません。

これらを使用して後にトレーニングを実行します。  
そのため、モデルに変更がある場合はケース分ファイルを作成する必要があります。

※ このモデルを読み込んでトレーニングを自動実行できるスクリプトを作成予定です。  
トレーニングのパラメータは、固定化されているためパラメータ化しやすいです。

編集方法
--

モデル変更のためには、一部コードを修正する必要があります。

モデルの内容は全て json ファイルに出力されますが、心配であれば Diff 等を取るとエビデンスになるかもしれません。  
私はそこまではしないつもりですが……

### ユースケース(出力クラス数)により変更が必要な部分

以下の部分を利用して出力層の次元数をコントロールしているため、その他データの有無によっても変更する必要があります。

面倒ですみません。

```
# 分類クラス定義
classes = ['male', 'female']
nb_classes = len(classes)
```

モデル定義では `nb_classes` 変数のみしか使用していません。  
ただ、同様の定義をトレーニング時に指定するため、クラス名まで指定してもらえると良いと思います。

### トレーニングケースにより変更が必要な部分

シーケンシャルモデル作成のコードが対象になります。

```
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])
```

#### 活性化関数

活性化関数を変更するときは、出力層の活性化関数 (`softmax`) 以外を `sigmoid` 等に変更して下さい。

利用可能な活性化関数は Keras の日本語ドキュメントにも記載があります。  
[https://keras.io/ja/activations/](https://keras.io/ja/activations/)

#### ドロップアウト

各層のドロップアウトは、上記では 0.25, 0.25, 0.5 に設定されています。  
この部分をそれぞれケースに合わせ変更して下さい。

今回のケースでは以下の 4 項目です。

1. 0.125, 0.125, 0.25
2. 0.25, 0.25, 0.25
3. 0.25, 0.25, 0.5
4. 0.5, 0.5, 0.75

#### 最適化関数

最適化関数の変更は 1 箇所のみの変更です。

`model.compile` の `optimizer` の値を `sgd` など変更して下さい。

利用可能な最適化関数は Keras の日本語ドキュメントに記載があります。
[https://keras.io/ja/optimizers/](https://keras.io/ja/optimizers/)

今のところは以下の 2 種類です。

1. RMSprop
2. sgd


複数の条件でトレーニングを一度に実行する
--

`train.sh` を用いて複数のトレーニングを一度に実行する。  
`train_args.txt` を参照してパラメータを設定し、`train_by_json.py` を実行する。

### train_args.txt

`train.sh` を実行するときに利用するパラメータの設定シート。  
タブ区切りで以下を設定する。

**※ 必ず最後に改行を挿入すること。改行がない行の設定では実行されない。**

設定値

|#|設定される値|備考|
|-:|:-|:-|
|1|model|読み込むモデルのパス|
|2|classes|分類クラスをカンマ区切りで設定|
|3|optimizer|最適化関数|
|4|targetsize-width|トレーニング時の画像の横幅. モデルの Input と合わせなくてはならない。|
|5|targetsize-height|トレーニング時の画像の縦幅. モデルの Input と合わせなくてはならない。|
|6|batchsize|トレーニングのミニバッチサイズ|
|7|epoch|トレーニングを実行する epoch 数|
|8|rotation-range|トレーニングデータの前処理. 画像をランダムに回転する範囲を浮動小数点数で設定. デフォルトは `0` を設定する|
|9|width-shift-range|トレーニングデータの前処理. ランダムに水平シフトすつ範囲浮動小数点数で設定. デフォルトは `0` を設定する|
|10|height-shift-range|トレーニングデータの前処理. ランダムに垂直シフトすつ範囲を浮動小数点数で設定. デフォルトは `0` を設定する|
|11|shear-range|トレーニングデータの前処理. シアー強度（反時計回りのシアー角度（ラジアン））を浮動小数点数で設定. デフォルトは `0` を設定する|
|12|zoom-range|トレーニングデータの前処理. ランダムにズームする範囲を浮動小数点数で設定. デフォルトは `0` を設定する|
|13|horizontal-flip|トレーニングデータの前処理. 水平方向に入力をランダムに反転するかを真偽値で設定(TRUE or FALSE). デフォルトは `FALSE` を設定する|
|14|vertical-flip|トレーニングデータの前処理. 垂直方向に入力をランダムに反転するかを真偽値で設定(TRUE or FALSE). デフォルトは `FALSE` を設定する|

