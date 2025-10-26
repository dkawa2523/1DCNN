# Instructions for extending and maintaining the temperature prediction code

このファイルは、Codex‑5 によるコードの拡張や修正を容易にするためのガイドラインを提供します。
リポジトリ全体の構造や主要なモジュール、そして変更や新機能を追加する際のポイントをまとめました。

## リポジトリ構造

```
temperature_prediction/
├── config_train.yaml      # 学習時のハイパーパラメータとデータ設定
├── config_pred.yaml       # 推論時の設定
├── generate_synthetic_data.py  # 合成データ生成ツール
├── data/                  # CSV データ格納用（学習用とテスト用）
├── src/
│   ├── data_loader.py     # データ読み込み・シーケンス作成・正規化
│   ├── models.py          # MLP, CNN1D, LSTM, XLSTM のモデル定義
│   ├── train.py           # 学習ロジック（データの読み込み～訓練・検証・保存）
│   └── predict.py         # 学習済みモデルによる逐次推論
├── models/                # 学習済みモデルの保存先
├── logs/                  # 学習ログ
└── predictions/           # 推論結果の保存先
```

## カスタマイズの基本手順

1. **新しいモデルの追加**
   * `src/models.py` にクラスを追加してください。必ず `nn.Module` を継承し、コンストラクタと `forward(self, x)` メソッドを実装します。
   * 追加したモデルを `build_model()` の分岐に登録することで、`config_train.yaml` の `model.type` で呼び出せるようになります。

2. **ハイパーパラメータの調整**
   * 学習に関するパラメータ（隠れ層の次元数、エポック数、学習率など）は `config_train.yaml` の `model` セクションで指定します。
   * 推論時のパラメータやモデルディレクトリは `config_pred.yaml` で設定します。

3. **入力特徴量やターゲットの変更**
 * 新しい CSV データに応じて `config_train.yaml`/`config_pred.yaml` の `data.input_columns` と `data.target_columns` を変更してください。
 * 本プロジェクトの標準データセットはファイル名が `temp_{brine}_{heater}_{plasma}_{init_temp}.csv` という形式で、CSV 本体には時刻と温度列だけが含まれます。ブライン・ヒーター・プラズマ・初期温度の定数条件は `src/data_loader.py` の `parse_filename()` によってファイル名から抽出され、各行に追加されます。
 * 温度以外の物理量を予測する場合もこのリストを更新し、モデル定義や正規化がこれらの列を使うようになっていることを確認します。
 * `read_csv_files()` は `(path, DataFrame)` のタプルを返します。ファイルごとの情報が必要な場合は
   パスを利用してください。データフレームのみが必要なときは内包表記で取り出せます。
 * テスト時の評価も行いたい場合は `config_train.yaml` の `data.test_dataset_dir` にデータパスを設定し、
   必要なら `test_input_columns` / `test_target_columns` を追加してください（省略時は学習と同じ列を使用）。

4. **xLSTM の実装強化**
   * 現在 `XLSTMModel` は標準の LSTM 実装をラップしたプレースホルダです。
   * xLSTM では指数ゲートやスカラー／行列メモリなど特有のゲーティング機構を導入します【348973373501979†L592-L637】。
   * 参考文献や論文の図式を元に、`src/models.py` に新しいセル構造を実装し、`XLSTMModel` でそれを使用するように改修してください。
   * 変更箇所の例:
     - 新規クラス `XLSTMCell(nn.Module)` を作成し、指数ゲートや正規化を実装する。
     - `XLSTMModel` のコンストラクタで `nn.ModuleList` に複数の `XLSTMCell` を積み重ねる。
     - `forward()` で各セルに順に入力を渡し、最終ステートを全結合層で出力に変換する。

5. **物理インフォームド損失の追加**
   * 温度場の時間発展方程式や境界条件を損失関数に組み込むことで、物理的に整合した予測を行うことができます【476801875214683†L112-L141】。
   * `src/train.py` で損失計算部分をカスタマイズし、例えばエネルギー保存則などを定式化した追加項を加えてください。

6. **推論時の挙動の変更**
   * `src/predict.py` では、最初の `sequence_length` ステップには実測温度を使い、以降は予測温度をフィードバックしています。
   * フィードバック方法を変更したい場合は `predict_file()` 内のループを編集してください。
   * 例えば、複数ステップをまとめて予測するシーケンス予測モデルに変更する場合、入力シーケンスから直接 `n_steps` 分の温度を出力するモデル構造とデータセットを用意する必要があります。

7. **新しいメトリクスの追加**
   * 予測精度を定量化するために MAE や RMSE のほかに決定係数 R² などを計算したい場合、`src/train.py` と `src/predict.py` に計算ロジックを追加してください。

## コーディングの注意点

* モデル追加や前処理の変更の際は、既存のインタフェース（データ形状や引数）を崩さないよう注意してください。
* 新しい依存ライブラリを導入する場合は、README にインストール方法を追記してください。
* データサイズが大きくなる場合は、`DataLoader` の引数 `num_workers` を調整して I/O を高速化できます。
* 学習時には `train_log.json`/`train_log.csv` にエポックごとの履歴が保存され、
  `loss_curve.png` と `validation_metrics.png` が生成されます。解析やレポート作成時に活用してください。
* 依存パッケージの追加・更新時は `requirements.txt` を更新し、`uv pip install -r requirements.txt`
  で整合性を確認してください。
* `training.cross_validation.num_folds` を 2 以上にすると `logs/cv/` 以下にフォールド別の学習ログと
  散布図、`cv_metrics.(json|csv)`、`cv_metrics.png`、`scatter_overall.png` が出力されます。
* 推論時には各 CSV ごとに真値‐予測値散布図 `scatter_*.png` が出力されます。R² と理想線が描画されるため、
  再現性のチェックに利用できます。
* 学習完了後は `test_metrics.(json|csv)` と `test_scatter/` ディレクトリが生成され、テストデータに対する
  真値・予測の散布図と R² が保存されます。

このガイドに従って、コードの拡張や保守を行ってください。不明点がある場合は README を参照し、コメントを付加することで将来の開発者の理解を助けてください。
