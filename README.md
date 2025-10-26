# 温度予測モデル構築プロジェクト

このプロジェクトは、導体エッチング装置の Cell パーツにおける温度分布を学習し、
ヒーターやブラインの制御条件およびプラズマからの入熱に基づいて計測点の温度を予測するための
機械学習パイプラインを提供します。CAE 解析から得られる CSV データを読み込み、
複数の深層学習モデル（MLP、1D CNN、LSTM、および拡張 LSTM(xLSTM)）を用いて
次時刻の温度を推定し、推定値をフィードバックしながら全時系列を再現します。

本プロジェクトで扱う学習用・推論用の CSV ファイルは
`temp_{brine}_{heater}_{plasma}_{init_temp}.csv` という命名規則に従います。
ファイルの内容にはヘッダー行がなく、1 列目に時刻、2 列目以降に各計測位置の
温度のみが含まれています。ブライン設定値(`brine`)、ヒーター出力(`heater`)、
プラズマ入熱(`plasma`)、初期温度(`init_temp`)といった条件はファイル名にのみ
記録されており、データローダーがファイル名からこれらの定数条件を抽出して
特徴量に含めます。例えば `temp_11.0_66_5000_140.csv` は、ブライン値 11.0、ヒーター出力 66、
プラズマ入熱 5000、初期温度 140 °C の条件でシミュレーションされた温度時系列を表します。

## 背景

CAE 計算では、装置の設定条件・ヒーターとブラインの値・プラズマの入熱パラメータ・
初期温度などを入力として、Cell 内各計測点の温度を微小時間ステップごとに計算します。
本プロジェクトではこれらのデータを学習用 CSV として取り込み、機械学習モデルで
逐次予測を行うことで長時間の温度履歴を高速に再現することを目指します。

### なぜ LSTM や xLSTM を利用するのか

従来の多層パーセプトロン（MLP）は非線形問題に適用できるものの時系列の依存性を直接扱えません。
一方でリカレントニューラルネットワーク(RNN)やその改良である LSTM は、隠れ状態を通じて
過去の情報を保持し時間依存性を表現できます【718251499221402†L815-L839】。
1D CNN は時系列データに畳み込みを適用することで局所的なパターンを抽出できますが、
長期の依存性を捉えるのは難しい場合があります【718251499221402†L792-L808】。
実際に水冷チラーの予測では、LSTM が MLP と 1D CNN よりも高い精度を示したものの
計算時間は長いという報告があります【718251499221402†L1075-L1082】。
最近提案された Extended LSTM (xLSTM) は、LSTM が長系列で重要な情報を忘れる問題を
解決するために指数ゲートやスカラー／行列メモリを導入し、メモリ容量と学習の安定性を
向上させています【348973373501979†L592-L637】。xLSTM を時系列予測へ応用した
xLSTMTime では系列分解やリバーシブル正規化を組み合わせることで他のモデルより優れた性能を示し、
改良されたゲーティング機構が長期的依存を捉えるのに有効であることが報告されています【348973373501979†L592-L606】。
さらに、物理法則を損失関数に組み込むことで高精度かつ長期予測を実現する
物理インフォームド機械学習の研究も進んでおり、
幾何学情報と物理的拘束を組み合わせた ConvLSTM モデルは 4.5〜13.9% の誤差で長時間の温度場を
予測できることが示されています【476801875214683†L112-L141】。

本プロジェクトではこれらの知見を踏まえ、LSTM を基本構造としつつ
xLSTM などへの拡張が容易なコード構造を採用しています。

## ディレクトリ構成

```
temperature_prediction/
├── README.md              # このファイル
├── instructions.md        # Codex-5 用の拡張指示
├── config_train.yaml      # 学習用設定ファイル
├── config_pred.yaml       # 推論用設定ファイル
├── generate_synthetic_data.py  # 合成データ生成スクリプト
├── data/
│   ├── train/             # 学習用 CSV データ
│   └── test/              # 評価用 CSV データ
├── models/                # 学習済みモデルと正規化統計値が保存されます
├── predictions/           # 推論結果の CSV が保存されます
├── logs/                  # 学習ログ（任意）
└── src/
    ├── data_loader.py     # データ読込・シーケンス構築・正規化
    ├── models.py          # 各種モデル定義
    ├── train.py           # 学習スクリプト
    └── predict.py         # 推論スクリプト
```

## セットアップ

1. **依存パッケージのインストール**

    Python 3.9 以降を想定しています。依存関係は [uv](https://github.com/astral-sh/uv) を利用すると効率的です。

    ```bash
    # uv が未導入の場合はインストール
    curl -Ls https://astral.sh/uv/install.sh | sh

    # 仮想環境を作成して有効化
    uv venv .venv
    source .venv/bin/activate      # Windows の場合は .venv\Scripts\activate

    # 必要なライブラリを一括インストール
    uv pip install -r requirements.txt
    ```

    既存の環境を使う場合も `uv pip install -r requirements.txt` を実行して、PyTorch / NumPy / Pandas /
    PyYAML / tqdm / matplotlib などの依存パッケージを揃えてください。

2. **合成データの生成（任意）**

    サンプルとして利用可能な合成データ生成スクリプト `generate_synthetic_data.py` を用意しています。
    このスクリプトには 2 種類のデータ生成方法があります。

    - **ランダムな制御プロファイルを使った小規模データ**: 旧形式の `generate_synthetic_data()` 関数は
      いくつかのトレーニング条件およびテスト条件をランダムに生成し、ヘッダー付きの CSV を出力します。
      本プロジェクトのデータ形式とは異なるため、新規データセットでの利用には推奨しませんが、
      動作確認用に残しています。

    - **定数条件を組み合わせたデータセット**: 推奨される方法は `generate_combination_dataset()` を使用し、
      ブライン値・ヒーター値・プラズマ入熱・初期温度の全組み合わせに対して
      `temp_{brine}_{heater}_{plasma}_{init_temp}.csv` というファイル名の CSV を生成するものです。
      CSV 内にはヘッダーがなく、時刻と温度列のみが保存されます。例えば以下のように
      Python から呼び出すことでデータを作成できます。

    ```python
    from generate_synthetic_data import generate_combination_dataset

    # 例: ブライン4種×ヒーター4種×プラズマ4種×初期温度4種 = 256 条件
    brine_vals = [0.4, 5.7, 11.0, 17.0]
    heater_vals = [0, 33, 66, 100]
    plasma_vals = [0, 5000, 10000, 15000]
    init_temps = [120, 140, 160, 180]

    # 学習用データを生成
    generate_combination_dataset(out_dir="data/train",
                                 brine_values=brine_vals,
                                 heater_values=heater_vals,
                                 plasma_values=plasma_vals,
                                 init_temps=init_temps,
                                 num_steps=200,
                                 n_points=5)

    # テスト用データを生成（必要に応じて別ディレクトリに保存）
    generate_combination_dataset(out_dir="data/test",
                                 brine_values=brine_vals,
                                 heater_values=heater_vals,
                                 plasma_values=plasma_vals,
                                 init_temps=init_temps,
                                 num_steps=200,
                                 n_points=5)
    ```

    上記の例では 256 個の CSV ファイルが `data/train` に作成され、同じ組み合わせの条件が
    `data/test` にも出力されます。学習と評価でデータを分けたい場合は、特定の組み合わせのみ
    テスト用ディレクトリに生成するか、生成後にファイルを振り分けてください。

3. **設定ファイルの編集**

    モデルやデータに関するパラメータは `config_train.yaml` と `config_pred.yaml` に記述します。
    - `dataset_dir`: CSV が置かれたディレクトリ。
    - `test_dataset_dir`: 学習完了後にモデルを評価するテスト CSV のディレクトリ（省略時はテスト評価をスキップ）。
    - `input_columns`: 入力特徴量となる列名。ファイル名から抽出される定数条件
      (`brine`、`heater`、`plasma`、`initial_temp`) と、予測に利用する過去の温度列を含めます。
    - `target_columns`: モデルが予測すべき温度列名。通常は各計測点の温度列を指定します。
    - `sequence_length`: モデルへ入力する過去時系列の長さ。LSTM や CNN を使う場合は 1 より大きくします。
    - `model.type`: `mlp`、`cnn1d`、`lstm`、`xlstm` のいずれかを指定します。
    - `model` 以下の値: 隠れ層の次元数や学習率など。
    - `training.cross_validation`: 交差検証の設定。`num_folds` を 2 以上にすると学習前に K-fold CV が動作し、
      各フォールドの学習ログ・散布図・指標が `logs/cv/` に出力されます。`shuffle` と `random_seed`
      でインデックスのシャッフル方法を制御できます。

## 学習の実行

以下のコマンドで学習を開始します。学習済みモデルは `output.model_dir` で指定したディレクトリに保存されます。

```bash
python3 -m src.train --config config_train.yaml
```

実行すると、データの読み込み→正規化→学習と検証が行われます。検証損失が改善した時点で
`best_model.pth` と正規化統計 `norm_stats.json` が `models` ディレクトリに保存されます。
学習履歴は `train_log.json` / `train_log.csv` に記録され、学習曲線
(`loss_curve.png` と `validation_metrics.png`) は `output.logs_dir` が設定されていればそのディレクトリ、
未指定の場合は `models` ディレクトリに生成されます。

`training.cross_validation.num_folds` を 2 以上に設定している場合は、学習前に K-fold 交差検証が実行されます。
各フォールドの学習ログと散布図 (`logs/cv/fold_i/` 以下) に加えて、フォールド毎の評価指標をまとめた
`cv_metrics.json` / `cv_metrics.csv`、および `cv_metrics.png` と `scatter_overall.png` が `logs/cv/` に出力されます。
どのフォールドでも真値と予測値の散布図に R² と理想線 (y=x) が描画されます。

さらに `data.test_dataset_dir` を設定している場合は、学習後にテストデータで逐次推論を行い、
各 CSV について真値と予測値の散布図 (`test_scatter/scatter_<ファイル名>.png`) と
全体の散布図 (`test_scatter/scatter_overall.png`) を生成します。テスト指標は
`test_metrics.json` および `test_metrics.csv` にエクスポートされるため、学習完了直後に
推論品質を確認できます。

## 推論の実行

学習したモデルを用いて新しい条件の温度時系列を予測するには、以下を実行します。

```bash
python3 -m src.predict --config config_pred.yaml
```

`config_pred.yaml` にはテストデータディレクトリやモデル保存先を指定します。
推論スクリプトは各 CSV ファイルに対して逐次的に予測を行い、結果を
`prediction_dir` に保存します。温度列が CSV に含まれている場合は
MAE と RMSE を表示し、真値と予測値の散布図 `scatter_<ファイル名>.png` を生成して
R² と理想線 (y=x) を併記します。

## モデルの仕組み

### 入力・出力

* **入力特徴量**: ファイル名に含まれる定常的な条件 (`brine`、`heater`、`plasma`、`initial_temp`)
  と、過去 `sequence_length` ステップ分の計測点温度。
* **出力**: 時刻 t+1 の各計測点温度。

モデルはこの「次時刻予測」を繰り返し適用して長期の温度変化を推定します。推論時には、
最初の sequence_length ステップだけ実測値を使い、以降はモデル予測値をフィードバックします。

### MLP

多層パーセプトロン (MLP) は入力系列をフラットにして全結合層に通す簡潔な構造です。
非線形性を表現できますが時系列の依存性を直接学習する仕組みがないため、
短いウィンドウでの近似に適します【718251499221402†L742-L764】。

### 1D CNN

一次元畳み込みニューラルネットワークは時間軸方向に畳み込みをかけ、局所的なパターンを
捉えます【718251499221402†L792-L808】。畳み込み層とプーリング層により特徴抽出を行い、
全結合層で出力を生成します。長期間の依存性を捕まえたい場合は、
カーネルサイズや層を調整する必要があります。

### LSTM

リカレントニューラルネットワークの一種である LSTM は、忘却ゲート・入力ゲート・出力ゲートを
持ち、長期依存関係を持つ時系列を効果的に学習できます【718251499221402†L815-L839】。
水冷チラーの電力消費予測の例では LSTM が MLP や 1D CNN より精度が高いことが報告されていますが、
計算時間は長くなる傾向があります【718251499221402†L1075-L1082】。

### xLSTM (拡張 LSTM)

xLSTM は従来の LSTM の欠点を改善するために提案された拡張版です。
指数ゲートやスカラー／行列メモリによって重要な情報の保持能力を高め、
長系列でも安定して学習できるように設計されています【348973373501979†L592-L637】。
本コードでは `XLSTMModel` クラスを標準の LSTM を基にしたプレースホルダとして実装しており、
クラス構造は xLSTM に置き換えやすいようになっています。

## テストデータと検証

`generate_synthetic_data.py` により生成される合成データは小規模なシステムであり、
機械学習パイプラインの動作確認に利用できます。推論時にはモデルが正しく
逐次的に予測値をフィードバックしているかを確認してください。

## 今後の拡張

* **物理インフォームド損失**: 温度のエネルギーバランス方程式や境界条件を損失関数に組み込む
  物理インフォームド機械学習(PINN)によって、長期予測の精度向上が期待できます【476801875214683†L112-L141】。
* **幾何学情報の活用**: Cell の形状や測温点の空間配置を特徴量に含めることで、
  プラズマ入熱の空間的非一様性を捉えやすくなります。
* **xLSTM の実装**: `src/models.py` の `XLSTMModel` はプレースホルダです。
  `instructions.md` に記載の手順に従って指数ゲートや行列メモリを導入することで、
  長期依存性に対する性能を向上させられます。

以下は今回のプロジェクトを実行するための仮想環境（uv利用）の構築手順と必要ライブラリのインストール、学習および推論の実行コマンドです。Unix系シェル（bash/zsh）を想定しています。Windowsの場合はパスの区切りや環境変数の記法を読み替えてください。

⸻

1. uvを利用した仮想環境の構築
	1.	uvのインストール（未導入の場合のみ）
uv は高速なPythonパッケージマネージャ兼仮想環境ツールです。pipx 経由でインストールできます。

pipx install uv
# または pip 経由で:
pip install uv


	2.	仮想環境の作成
プロジェクトルートで以下を実行すると .venv ディレクトリに仮想環境が作成されます。特定のPythonバージョンを指定したい場合は --python オプションでバージョンを指定します。

cd /path/to/project  # プロジェクトのルートへ移動
uv venv .venv        # Pythonのバージョンを自動検出して仮想環境作成
# 例: Python3.11を使う場合
# uv venv --python 3.11 .venv


	3.	仮想環境の有効化

# macOS/Linuxの場合
source .venv/bin/activate

# Windowsの場合
.venv\Scripts\activate

有効化するとプロンプトに (venv) が表示され、以降のコマンドは仮想環境上で実行されます。

⸻

2. 必要ライブラリのインストール

本プロジェクトで最低限必要なパッケージは torch、pyyaml、pandas、numpy、tqdm です。以下のコマンドでインストールします。

# 仮想環境内で実行
uv pip install torch pyyaml pandas numpy tqdm

必要に応じて、モデルの拡張や評価で追加ライブラリ（scikit-learn など）を使用したい場合は同様に uv pip install ライブラリ名 でインストールします。

⸻

3. データセットの準備（必要に応じて）

プロジェクトには既に学習用・テスト用データセットが temperature_prediction/data/train および temperature_prediction/data/test に含まれています。別の条件でデータセットを作成したい場合は Python インタプリタから generate_combination_dataset を呼び出してください。

python - <<'PY'
from temperature_prediction.generate_synthetic_data import generate_combination_dataset

# 任意の値リストを指定
brine_vals  = [0.4, 5.7, 11.0, 17.0]
heater_vals = [0, 33, 66, 100]
plasma_vals = [0, 5000, 10000, 15000]
init_temps  = [120, 140, 160, 180]

# 学習用データを生成
generate_combination_dataset(
    out_dir="temperature_prediction/data/train",
    brine_values=brine_vals,
    heater_values=heater_vals,
    plasma_values=plasma_vals,
    init_temps=init_temps,
    num_steps=50,    # 時間ステップ数。適宜変更可
    n_points=5       # 計測点数。適宜変更可
)

# テスト用データ（必要に応じて）
generate_combination_dataset(
    out_dir="temperature_prediction/data/test",
    brine_values=brine_vals,
    heater_values=heater_vals,
    plasma_values=plasma_vals,
    init_temps=init_temps,
    num_steps=50,
    n_points=5
物理インフォームドConvLSTM	ConvLSTMに幾何学情報と物理法則を組み込み、時空間温度分布を予測 ￼	損失に初期条件・境界条件・物理拘束を含め、少数データでも物理的に整合した結果を生成	4.5〜13.9%の誤差削減と50%の学習時間短縮を達成 ￼; 空間的な相互作用を捉えられる	物理モデルの導入が必要で設計が複雑; データドリブンのみのモデルより実装コストが高い	渦状のプラズマ加熱や伝熱方程式を満たす温度分布予測に非常に適しており、データ不足を補える	金属積層造形の温度場予測や流体シミュレーションで実績 ￼	PINNやGNNとの融合、異なる物理現象の同時学習など拡張の余地が大きい
Transformer系モデル	自己注意機構を用いたエンコーダ–デコーダ。長距離依存性を並列に学習	マルチヘッド注意により複数の相関を同時に捉える; 位置エンコーディングで時間順序を表現	長期パターンを効果的に抽出し高い精度; 計算は並列化可能	多数のパラメータと学習データが必要; 訓練時間が長い; 過学習しやすい	データが十分あれば温度時系列の複雑なパターンを捉えられる。小規模データでは過適合の懸念	建物エネルギー消費の予測で改良版TransformerがLSTMやGRUより低MAPEを達成 ￼; 気象予測や金融時系列にも応用	AutoformerやInformerなど時系列特化の改良が進んでおり、注意機構と物理知識の組み合わせも研究中
物理インフォームドグラフニューラルネットワーク (GNN)	センサ位置に基づく非対称グラフを構築し、グラフ注意・グラフゲートで空間–時間特徴を抽出 ￼	物理的な非対称関係を捉える隣接行列; 空間的結合と時間的ゲートの融合 ￼	センサ間の関係をモデル化しMAE 0.75 °C未満の高精度を達成 ￼	グラフ構造設計や位置情報が必要; モデルが複雑で計算負荷が大きい	セル内の測温点の空間配置と相互作用を考慮することでより高精度な温度予測が期待できる	南中国の気温予測 ￼、海面温度予測、都市ヒートウェーブ予測など	動的グラフやトポロジ学習との統合、Transformerと組み合わせたグラフ注意機構などが発展中

他の手法の可能性
	•	統計モデル: ARIMAや季節ARIMAは線形な依存関係を前提とした統計的時系列手法で、短期予測や単変量データに適している。非線形性や外部変数が強い本問題では単独では不十分だが、誤差補正用に併用できる ￼。
	•	ハイブリッド・エンコーダ–デコーダ: CNNで局所的特徴を抽出しLSTMやTransformerで時系列をモデリングする CNN–LSTM や CNN–Transformer は複雑なパターンを捉えやすく、多変量エネルギー予測でも実績がある ￼ ￼。
	•	PINN/機械学習と物理モデルの統合: 本問題では熱伝導方程式や境界条件を損失に組み込んだモデルが有効。ConvLSTM や GNN だけでなく、PINN を LSTM や Transformer と組み合わせる研究が進んでいる ￼ ￼。

これらのモデルを比較しながら、利用可能なデータ量、計測点の空間的配置、計算資源、予測精度の要求などを考慮して最適な手法を選択することが重要です。

## 引用文献

* LSTM の構造とゲーティング機構について: 【718251499221402†L815-L839】。
* MLP と 1D CNN の概要: 【718251499221402†L742-L764】【718251499221402†L792-L808】。
* LSTM が MLP および 1D CNN よりも高精度だが計算時間が長いという報告: 【718251499221402†L1075-L1082】。
* xLSTM の改良ポイントと長期依存性への対応: 【348973373501979†L592-L637】。
* 物理インフォームド ConvLSTM による長期温度予測の成果: 【476801875214683†L112-L141】。
