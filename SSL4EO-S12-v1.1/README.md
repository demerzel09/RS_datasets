# MS-CLIP ( マルチスペクトル CLIP )用データセット - SSL4EO-S12 v1.1　より抜粋

* コード : https://github.com/IBM/MS-CLIP
* データセット : 
    * 主にSentinel-2 で構成　SSL4EO-S12 v1.1
    * 2.13 TB と暴対のため、validation用 8G を利用する
    * （内部用リンク）dataset_2025_11_1: SSL4EO-S12 v1.1 ( MS-CLIP のデータセット）

* 自己教師あり学習・教師なし用の Sentinel-2  (マルチスペクトル と Sentinel-1 (SAR) のデータセット。ラベルなし。

https://github.com/zhu-xlab/SSL4EO-S12

* 世界の上位１万都市＋その周辺50 km範囲をカバー、４季節を跨ぐ時相データを取得 
* 構成：
    * S2L1C：Sentinel‑2 のレベル1C製品（Top-of-Atmosphere (TOA) 反射率、13バンド）です。
    * S2L2A：Sentinel-2 のレベル2A製品（Surface Reflectance、大気補正済、12バンド）です。
    * S1GRD：Sentinel‑1 の「GRD（Ground Range Detected）／グラウンドレンジ検出」製品を指しています。技術報告書中に「Modality DType Range Units #Bands: S1 GRD float16 -50 – +1 dB 2」などの記述があり、2バンド（VV, VH）を含むSARデータであることが明記されています。 
    * S2RGB：Sentinel-2データから「RGB（赤・緑・青）3バンド」に圧縮・変換したものを意味しています。技術報告書では “we provide S2 RGB data based on L2A products” とあり、反射率データ（12バンド）を8ビットRGB画像として可視化用途／簡便用途向けに変換した
* MS-CLIP はこのデータセットに Llama3‑SSL4EO‑S12‑v1.1‑captions というデータセットを対応させてファインチューニングしている。

https://huggingface.co/datasets/ibm-esa-geospatial/Llama3-SSL4EO-S12-v1.1-captions

* 各サンプルは「ひとつの位置（センター座標）＋4つの時刻（４季節分）＋モダリティ（S2 L1C／S2 L2A／S2 RGB／S1 GRD）」一つの画像(sample_id) に対し、以下のラベルデータがある。
    * ４つの季節
    * 関連クラス・タグ
    * 地形の名前
    * 質問 1/2/3
    * caption


## データセット入手  
  https://github.com/zhu-xlab/SSL4EO-S12  

  のReadmeの 下記のリンクより入手
  Example subset: An example 100-patch subset (600MB) is available at Google Drive.  

  下記は大きすぎて、この案件では使っていないが、以下にもある
  https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/tree/main/val/S2L2A


SSL4EO-S12-v1.1/
├── README.md
└──ssl4eo-s12_100patches
    ├── rgb
    ├── s1
    ├── s2a
    └── s2c
