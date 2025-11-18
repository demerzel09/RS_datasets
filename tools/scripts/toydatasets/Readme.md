# tools ディレクトリの使い方

本フォルダには、リモートセンシング用のデータ生成・変換スクリプトを収録しています。各スクリプトの概要・前提条件・実行例をまとめます。

## make_toy_dataset.py（トイデータ作成）
- 概要: ワークスペース内の複数データセットから小規模な学習/検証用データセットを作成し、JSONL と正規化済み画像を出力します。
- 出力: `caption.jsonl`, `vqa.jsonl`, `ref.jsonl`, `cls.jsonl`, `paired.jsonl(任意)`, `vqa_ts.jsonl(任意)` と、`images/` 配下の画像群。
- 主なオプション（None を指定/省略すると該当タスクはスキップ）
  - `--root .` ルートディレクトリ（既定: カレント）
  - `--out toy_datasets` 出力先ディレクトリ
  - `--seed 42` 乱数シード
  - `--resize 512` 画像の長辺ピクセル
  - `--jpeg` PNG の代わりに JPEG 保存
  - `--count-caption` Caption 件数（指定なし、None でスキップ）
  - `--count-vqa` VQA 画像数（指定なし、None でスキップ、1画像の全Q&Aを含める）
  - `--count-ref` Referring 画像数（指定なし、None でスキップ、1画像の全表現を含める）
  - `--count-cls` 分類画像数（指定なし、None でスキップ）
  - `--make-paired` SSL4EO-S12 のペア/RAW も作成　（指定なしでスキップ）
  - `--count-vqa-ts` 時系列VQAのシリーズ数（指定なし、None でスキップGeoLLaVA/SSL4EO を混在）
  - `--vqa-ts-frames` シリーズあたりのフレーム数
- 実行例:
```
python tools/make_toy_dataset.py \
  --root . --out toy_datasets --seed 42 \
  --count-caption 300 --count-vqa 200 --count-ref 50 --count-cls 200 \
  --make-paired --count-vqa-ts 20 --vqa-ts-frames 4
```
## convert_landslide4sense_h5.py（Landslide4Sense 変換）
- 概要: Landslide4Sense-2022 の HDF5 を GeoTIFF/PNG に変換します。
- 出力: 各サンプルごとに以下を作成します。
  - `BAND.tif`: Sentinel‑2 のみ（B1–B12）の 12 バンド（(H,W,C) マルチバンド）
  - `SLOPE.tif`: 傾斜（B13 または専用データセット）
  - `DEM.tif`: 標高（B14 または専用データセット）
  - `MASK.tif`: マスク（`split/mask/mask_XXX.h5` がある場合）
  - `RGB.png`: B4/B3/B2 を min-max 正規化した可視化画像
  - `mask.png`: マスクの 8bit グレースケール画像（0/1 → 0/255 に拡大）
- 依存: `pip install h5py numpy tifffile pillow`（必要に応じて）
- 主なオプション
  - `--root Landslide4Sense-2022` ルートパス
  - `--split TestData|ValidData|Both` 変換対象
  - `--out out_landslide_s2` 出力ルート
  - `--ext .tif` 出力拡張子（`.tif` 推奨）
  - `--verbose` ログ多め
- 実行例:
```
python tools/convert_landslide4sense_h5.py \
  --root Landslide4Sense-2022 --split Both \
  --out out_landslide_s2 --ext .tif --verbose
```

## データセット配置ヒント（フォルダ構造）
- make_toy_dataset.py が参照する想定の配置例（最低限）
  - `rs5m_test_data/rsicd/`
    - `RSICD_images/*.jpg`
    - `rsicd_test.csv`（列: `filename`, `title` など）
  - `rs5m_test_data/rsitmd/`
    - `images/*.jpg`
    - `rsitmd_test.csv`（列: `filename`, `title` など）
  - `VRSBench/`
    - `VRSBench_EVAL_Cap.json`
    - `VRSBench_EVAL_vqa.json`
    - `VRSBench_EVAL_referring.json`
    - `Images_train/<image_id>.png` または `Images_val/<image_id>.png`
  - `rs5m_test_data/AIR-SLT/`
    - `imgs/*.jpg`
    - `annotations/anno.json`（points: 多角形座標）
  - 分類データ（任意）
    - `rs5m_test_data/AID/<class_name>/*.jpg`
    - `rs5m_test_data/RESISC45/<class_name>/*.jpg`
    - `rs5m_test_data/EuroSAT/<class_name>/*.jpg`（存在する場合）
  - SSL4EO-S12 100パッチ（任意: paired / 時系列VQA 用）
    - `SSL4EO-S12-v1.1/ssl4eo-s12_100patches/`
      - `rgb/<patch_id>/*.png`（ファイル名に日付含む）
      - `s2a/<patch_id>/<YYYYMMDD>/B02.tif, B03.tif, ...`（生バンド）
      - `s2c/<patch_id>/<YYYYMMDD>/Bxx.tif`（あれば）
      - `s1/<patch_id>/<YYYYMMDD>/VV.tif, VH.tif`
  - GeoLLaVA（任意: 時系列VQA 用）
    - `GeoLLaVA/updated_val_annotations.json`（conversations を含む）
    - `GeoLLaVA/updated_val_videos/*.mp4`

- convert_landslide4sense_h5.py が参照する想定の配置例
  - `Landslide4Sense-2022/`
    - `TestData/`
      - `img/image_1.h5, image_2.h5, ...`
      - `mask/mask_1.h5, mask_2.h5, ...`（存在する場合）
    - `ValidData/`
      - `img/image_XXX.h5`
      - `mask/mask_XXX.h5`（存在する場合）
  - 備考: H5 内には 12 バンドの Sentinel‑2（B1–B12）と、B13=Slope, B14=DEM を含む 14 バンド構成のことがあります。

## 出力構造（フォルダ/ファイル例）
- make_toy_dataset.py の出力（既定: `toy_datasets/`）
```
toy_datasets/
  caption.jsonl                # キャプション用アノテーション（指定時）
  vqa.jsonl                    # VQA 用アノテーション（指定時）
  ref.jsonl                    # リファリング用アノテーション（指定時）
  cls.jsonl                    # 分類用アノテーション（指定時）
  paired.jsonl                 # ペア/RAW（--make-paired 指定時）
  vqa_ts.jsonl                 # 時系列VQA（--count-vqa-ts > 0 のとき）
  summary.json                 # 生成サマリ
  logs/
    skipped.txt               # スキップ理由ログ
  images/
    caption/
      rsicd/<image>.png
      rsitmd/<image>.png
      vrsbench/<image_id>.png
    vqa/
      vrsbench/<image_id>.png
    ref/
      vrsbench/<image_id>.png
      air-slt/<image>.png
    cls/
      AID/<class>/<image>.png
      RESISC45/<class>/<image>.png
      EuroSAT/<class>/<image>.png
    paired/
      ssl4eo-s12/
        <patch>_<date>_rgb.png
        <patch>_<date>_s2rgb.png
        <patch>_<date>_sar.png
        raw/
          <patch>_<date>_s2a/B02.tif, B03.tif, ...
          <patch>_<date>_s2c/Bxx.tif (あれば)
          <patch>_<date>_s1/VV.tif, VH.tif
    vqa_ts/
      ssl4eo-s12/
        <patch>_<date>_rgb.png
        <patch>_<date>_s2rgb.png (あれば)
        <patch>_<date>_sar.png (あれば)
        raw/
          <patch>_<date>_s2a/B02.tif, B03.tif, ...
          <patch>_<date>_s2c/Bxx.tif (あれば)
          <patch>_<date>_s1/VV.tif, VH.tif (あれば)
      geollava/
        <video_stem>_00_rgb.png, <video_stem>_01_rgb.png, ...
```

- convert_landslide4sense_h5.py の出力（例: `out_landslide_s2/`）
```
out_landslide_s2/
  TestData/
    image_1/
      BAND.tif    # 12バンド (B1–B12, (H,W,C))
      SLOPE.tif   # 単バンド (B13)
      DEM.tif     # 単バンド (B14)
      MASK.tif    # マスク（存在時）
      RGB.png     # B4/B3/B2 の可視化
      mask.png    # MASK.tif の 8bit 可視化
    image_2/
      ...
  ValidData/
    image_XXX/
      ...
```

## 備考
- QGIS での表示を考慮し、BAND.tif は (H,W,C) のインターリーブ形式で保存します。
- `make_toy_dataset.py` の `--count-*` を省略した場合、そのタスクは実行されません（スキップ）。
