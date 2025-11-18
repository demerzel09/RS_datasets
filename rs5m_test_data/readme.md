
# GeoRSCLIP  ( RS特化 CLIP ) - RS5Mの評価のみ  
  データセットが豊富、Tellus AI Playground の基盤モデルとして利用している実績あり。  
  ただしRGB画像のみの扱いとなっている。  

  * コード :   
    https://huggingface.co/Zilun/GeoRSCLIP/tree/main
  * データセット : Github に付属
  * （内部用リンク）dataset_2025_11_1: rs5m_test_data ( GeoRSCLIP用 )  
       https://huggingface.co/Zilun/GeoRSCLIP

```
git clone https://huggingface.co/Zilun/GeoRSCLIP
cd GeoRSCLIP

Unzip the test data
unzip data/rs5m_test_data.zip
```
rs5m_test_data/  
├── AID  
│   ├── Airport  
│   ├── Bare Land  
│   ├── Baseball Field  
│   ├── Beach  
│   ├── Bridge  
・・・・   
│   ├── Square  
│   ├── Stadium  
│   ├── Storage Tanks  
│   └── Viaduct  
├── AIR-SLT  
│   ├── annotations  
│   ├── imgs  
│   └── selo_cache  
├── RESISC45  
│   ├── airplane  
│   ├── airport  
│   ├── baseball diamond  
│   ├── basketball court  
・・・・   
│   ├── tennis court  
│   ├── terrace  
│   ├── thermal power station  
│   └── wetland  
├── eurosat-rgb  
│   ├── 2750  
│   ├── eurosat  
│   └── split_zhou_EuroSAT.json  
├── rsicd  
│   ├── RSICD_images  
│   ├── dataset_rsicd.json  
│   ├── make_rsicd_dataset.py  
│   ├── rsicd_test.csv  
│   └── split_files  
└── rsitmd
    ├── dataset_RSITMD.json  
    ├── images  
    ├── make_rsitmd_dataset.py  
    ├── rsitmd_api.py  
    ├── rsitmd_test.csv  
    └── split_files  