#!/usr/bin/env python3
"""
簡易スクリプト: capture-first スキーマのサンプルJSON群を作成する
このスクリプトは既存の `make_toy_dataset.py` を上書きしないよう、別名で作成しています。

使い方（PowerShell）:
python .\tools\src\make_samples_capture_series.py

生成先: `samples/` に `vqa.json`, `vqa_ts.json`, `paired.json`, `caption.json` を出力します。
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "samples"

def write_sample(name, data):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")

def main():
    # vqa.json
    vqa = {
        "info": {"name": "vqa_sample", "version": "v1"},
        "questions": [
            {"q_id": "q_0001", "question": "この建物は何ですか？"},
            {"q_id": "q_0002", "question": "屋根の色は何色ですか？"}
        ],
        "captures": [
            {
                "capture_id": "c_20210801_35.1_139.7",
                "timestamp": "2021-08-01T10:00:00Z",
                "location": {"lat": 35.1, "lon": 139.7},
                "sensors": [
                    {"sensor_type": "rgb", "file_name": "rgb/20210801_0001.jpg"},
                    {"sensor_type": "sar", "file_name": "sar/20210801_0001.tif"}
                ],
                "captions": [{"text": "飛行機格納庫の外観"}],
                "objects": [{"object_id": "o1", "bbox": [100, 200, 300, 250], "category_id": 2}],
                "classifications": [{"category_id": 10, "score": 0.95}]
            },
            {
                "capture_id": "c_20210802_35.1_139.7",
                "timestamp": "2021-08-02T10:00:00Z",
                "location": {"lat": 35.1, "lon": 139.7},
                "sensors": [{"sensor_type": "rgb", "file_name": "rgb/20210802_0001.jpg"}],
                "captions": [{"text": "隣接する倉庫"}],
                "objects": [{"object_id": "o2", "bbox": [120, 210, 200, 180], "category_id": 3}],
                "classifications": [{"category_id": 11, "score": 0.88}]
            }
        ],
        "annotations": [
            {
                "capture_id": "c_20210801_35.1_139.7",
                "vqa_instances": [
                    {"q_id": "q_0001", "answers": ["ハンガー"]},
                    {"q_id": "q_0002", "answers": ["灰色"]}
                ]
            },
            {
                "capture_id": "c_20210802_35.1_139.7",
                "vqa_instances": [
                    {"q_id": "q_0001", "answers": ["倉庫"]}
                ]
            }
        ]
    }

    # vqa_ts.json
    vqa_ts = {
        "info": {"name": "vqa_ts_sample", "version": "v1"},
        "questions": [
            {"q_id": "q_v001", "question": "機体は写っていますか？"}
        ],
        "captures": [
            {"capture_id": "c_t0", "timestamp": "2021-08-01T10:00:00Z", "location": {"lat": 35.1, "lon": 139.7}, "sensors": [{"sensor_type": "rgb", "file_name": "rgb/t0.jpg"}]},
            {"capture_id": "c_t1", "timestamp": "2021-08-01T10:00:01Z", "location": {"lat": 35.1, "lon": 139.7}, "sensors": [{"sensor_type": "rgb", "file_name": "rgb/t1.jpg"}]},
            {"capture_id": "c_t2", "timestamp": "2021-08-01T10:00:02Z", "location": {"lat": 35.1, "lon": 139.7}, "sensors": [{"sensor_type": "rgb", "file_name": "rgb/t2.jpg"}]}
        ],
        "series": [
            {"series_id": "s_0001", "capture_ids": ["c_t0", "c_t1", "c_t2"], "meta": {"description": "短い時系列"}}
        ],
        "series_annotations": [
            {
                "series_id": "s_0001",
                "frame_annotations": [
                    {"capture_id": "c_t0", "vqa_instances": [{"q_id": "q_v001", "answers": ["はい"]}]},
                    {"capture_id": "c_t1", "vqa_instances": [{"q_id": "q_v001", "answers": ["いいえ"]}]},
                    {"capture_id": "c_t2", "vqa_instances": [{"q_id": "q_v001", "answers": ["はい"]}]}
                ]
            }
        ]
    }

    # paired.json
    paired = {
        "info": {"name": "paired_series_sample", "version": "v1"},
        "captures": [
            {"capture_id":"c_20210801_t0","timestamp":"2021-08-01T10:00:00Z","location":{"lat":35.1,"lon":139.7},"sensors":[{"sensor_type":"rgb","file_name":"rgb/20210801_t0.jpg"},{"sensor_type":"ms","file_name":"ms/20210801_t0.tif"}]},
            {"capture_id":"c_20210801_t1","timestamp":"2021-08-01T10:05:00Z","location":{"lat":35.1,"lon":139.7},"sensors":[{"sensor_type":"rgb","file_name":"rgb/20210801_t1.jpg"},{"sensor_type":"ms","file_name":"ms/20210801_t1.tif"}]},
            {"capture_id":"c_20210801_t2","timestamp":"2021-08-01T10:10:00Z","location":{"lat":35.1,"lon":139.7},"sensors":[{"sensor_type":"rgb","file_name":"rgb/20210801_t2.jpg"},{"sensor_type":"ms","file_name":"ms/20210801_t2.tif"}]},
            {"capture_id":"c_20210801_t3","timestamp":"2021-08-01T10:15:00Z","location":{"lat":35.1,"lon":139.7},"sensors":[{"sensor_type":"rgb","file_name":"rgb/20210801_t3.jpg"},{"sensor_type":"ms","file_name":"ms/20210801_t3.tif"}]}
        ],
        "series": [
            {"series_id": "s_locA_20210801", "capture_ids": ["c_20210801_t0", "c_20210801_t1", "c_20210801_t2", "c_20210801_t3"], "meta": {"location_name": "siteA"}}
        ]
    }

    # caption.json
    caption = {
        "info": {"name": "caption_sample", "version": "v1"},
        "captures": [
            {
                "capture_id": "c_20210801_35.1_139.7",
                "timestamp": "2021-08-01T10:00:00Z",
                "location": {"lat": 35.1, "lon": 139.7},
                "sensors": [{"sensor_type": "rgb", "file_name": "rgb/20210801_0001.jpg"}],
                "captions": [
                    {"text": "飛行機格納庫の外観", "source": "annotator_1"},
                    {"text": "大きな屋根と滑走路の一部", "source": "annotator_2"}
                ]
            },
            {
                "capture_id": "c_20210802_35.1_139.7",
                "timestamp": "2021-08-02T10:00:00Z",
                "location": {"lat": 35.1, "lon": 139.7},
                "sensors": [{"sensor_type": "rgb", "file_name": "rgb/20210802_0001.jpg"}],
                "captions": [{"text": "隣接する小さな倉庫", "source": "annotator_1"}]
            }
        ]
    }

    # write files
    write_sample("vqa.json", vqa)
    write_sample("vqa_ts.json", vqa_ts)
    write_sample("paired.json", paired)
    write_sample("caption.json", caption)

if __name__ == "__main__":
    main()
