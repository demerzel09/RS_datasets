# 衛星データセット操作用  
**Capture-First スキーマ説明**

- **目的**: 既存の `make_toy_dataset.py` のコレクタ出力（jsonl / 1行＝1画像）から冗長性を排し、画像依存ラベルを `capture` 単位に集約した「capture-first」形式の JSON を生成するためのスキーマ説明書。
- **対象ファイル**: ラッパー `make_toy_dataset_to_json.py` が出力するファイル:
  - `caption.json` — キャプション（captures に captions を含む）
  - `vqa.json` — VQA（`questions` 銀行 + `captures` + `annotations`）
  - `ref.json` — Referring（captures + refs）
  - `cls.json` — Classification（captures + categories）
  - `paired.json` — マルチモダリ / 時系列ペア（captures + series）
  - `vqa_ts.json` — 時系列 VQA（questions + captures + series + series_annotations）

**基本方針**
- 画像に依存するラベル（キャプション、アノテーション、分類ラベル、ポリゴン等）はすべて `capture` の中に収める。
- 質問（question）はテキスト→`q_id` の辞書（questions 銀行）で共通化し、各 `vqa_instances` は `q_id` を参照する。回答（answer）は capture 固有で保持する。
- 元の生データ（raw、たとえば Sentinel バンドの `.tif` や S1 の VV/VH）は `capture["sensors"]` に保持し、注釈で参照される画像は正規化済みの RGB を参照する。

------------------

## `capture` オブジェクト（共通）

例:
```
{
  "capture_id": "c_0000017_20200203_rgb",
  "timestamp": null,
  "location": {"lat": 35.1234, "lon": 139.5678},
  "sensors": [
    {"sensor_type": "rgb", "file_name": "images/paired/ssl4eo-s12/0000017_20200203_rgb.png"},
    {"sensor_type": "s2", "file_name": "images/paired/ssl4eo-s12/0000017_20200203_s2rgb.png"},
    {"sensor_type": "s2a_raw", "file_name": "images/.../raw/.../B2.tif", "tag": "10m"},
    {"sensor_type": "s1_raw", "file_name": "images/.../raw/.../VV.tif", "band": "VV"}
  ]
}
```

- `capture_id`: この JSON 内で一意の識別子（スクリプトはファイル名由来で `c_<stem>` を生成）。
- `timestamp`: 可能なら撮影日時（ISO 文字列推奨）を入れる。現在はソースに依存して未設定のことが多い。
- `location`: 可能なら `{lat, lon}` の辞書。元データの `location`/`lat`/`longitude` や `coords` から正規化して埋める。
- `sensors`: センサー別ファイル一覧。最低でも注釈は `sensor_type: "rgb"` の正規化画像ファイルを参照すること。
  - `sensor_type` の例: `rgb`, `s2`, `s1`, `s2a_raw`, `s2c_raw`, `s1_raw`。
  - raw グループ（例: `s2a_raw`）は `tag`（解像度ラベル, 例 `10m`/`20m`/`60m`）を付与する。
  - S1 raw は `band`（`VV`/`VH`）を付与する。

------------------

## タスク別フォーマット（主要）

- `caption.json`
  - 形式: `{ "info":..., "captures": [<capture objects with "captions": [...]>] }`
  - `capture` 内に `captions`: `[ {"text": "...", "source": "RSICD"}, ... ]` を入れる。

- `vqa.json`
  - 形式: `{ "info":..., "questions": [ {"q_id": "...", "question": "..."}, ... ], "captures": [...], "annotations": [...] }`
  - `questions`: 質問銀行。既存ソースに `question_id` があればそれを尊重する。もし同一 ID が異なるテキストで現れる場合はネームスペース化（`{source}_{qid}`）して衝突を回避する。
  - `annotations`: 各 capture ごとの VQA インスタンス: `{ "capture_id": "c_xxx", "vqa_instances": [ {"q_id": "...", "question": "...", "answers": [ ... ]}, ... ] }`
  - 注: アノテーションは RGB 画像を参照（`capture.sensors` 内の `sensor_type: "rgb"`）。raw ファイルは `sensors` に残すのみ。

- `ref.json`
  - 形式: `{ "info":..., "captures": [...], "refs": [ {"capture_id":..., "expression":..., "bbox":..., "polygon":..., "source":...}, ... ] }`
  - `capture` には `objects` 配列を入れる事も可能（現状はトップレベル `refs` で参照を持つ）。

- `cls.json`
  - 形式: `{ "info":..., "categories": [...], "captures": [...] }`
  - `capture` 内に `classifications`: `[ {"label": "...", "source": "AID"}, ... ]`

- `paired.json`
  - 形式: `{ "info":..., "captures": [...], "series": [ {"series_id": "s_<patch>", "capture_ids": [...], "meta": {...} }, ... ] }`
  - 各 `capture.sensors` に raw ファイル群（`s2a_raw`/`s2c_raw`/`s1_raw`）を追加して保存する。注釈は RGB を参照する。

- `vqa_ts.json`
  - 形式: `{ "info":..., "questions": [...], "captures": [...], "series": [...], "series_annotations": [...] }`
  - `series_annotations`: 各 series のフレーム単位の `vqa_instances` を `capture_id` 単位で持つ。

------------------

## question id の取り扱い（重要）
- 元ソースに `question_id`（または `qid`）がある場合は可能な限りその ID を `q_id` として保持します。
- 同じ `q_id` が異なるテキストで現れた場合は、ソース名でネームスペース化します（例: `VRSBench_12345`）。
- ソースに ID が無い問いは自動生成 ID（`q_0001` や `q_auto_<hash>`）を割り当てます。

------------------

## 実行例
PowerShell での短時間実行（テスト）:
```powershell
python .\tools\src\make_toy_dataset_to_json.py --root . --out toy_datasets_json_test --count-caption 5 --count-vqa 5 --make-paired --count-vqa-ts 2 --vqa-ts-frames 4
```

本番（既存 `make_toy_dataset.py` と同等の出力量）:
```powershell
python .\tools\src\make_toy_dataset_to_json.py --root . --out toy_datasets_json_production --count-caption 300 --count-vqa 200 --count-ref 50 --count-cls 200 --make-paired --count-vqa-ts 50 --vqa-ts-frames 4
```

------------------

## 補足・運用注意
- このスキーマは「注釈は正規化済み RGB を参照し、raw の正データは `sensors` に残す」方針を踏襲しています。
- `location` / `timestamp` の正規化は入力ソースに依存するため、必要に応じて `extract_location_from_row` のルール拡張を検討してください。
- README の内容や `sensor_type` 命名（`s2a_raw` 等）はプロジェクト内の運用規約に合わせて変更可能です。変更希望があれば反映します。

------------------

更新履歴:
- 2025-11-18: 初版（capture-first スキーマ説明、raw ファイル保存方針、question id ポリシー、実行例）
