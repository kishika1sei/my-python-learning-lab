# RAGチャットボットSampleApp

## アプリ作成の目的

- 個人学習用(RAGの基礎を実装する)
  - Web検索API(Serp)を使ったRAGの試用
  - Document(local)を読み込ませたRAGの試用

### アプリのコンセプト

- 国が提供している各種補助金、助成金関係のPDFは専門用語が多く一般の人にはわかりにくい→これをどうにかRAGの力でかみ砕いて説明できないか？という発想から個人学習も兼ねて作成に至った。

### 想定ユーザー

- 一般人、学生、地方自治体職員
  
### どんな機能を提供するのか

- ユーザが投入したPDFをもとにLLMが補助金関連の回答を行う
- Web検索を用いて、検索情報をもとにLLMが補助金関連の回答を行う
- Hybridモード(Doc/Web)の両方の情報を参照して回答する
- いずれの回答方法でも、AIがかみ砕いた用語で回答を行う


<!-- # セットアップ
pip install -r requirements.txt
cp .env.sample .env  # 値を設定
mkdir -p data/pdf -->

### 事前インデックス
<!-- python scripts/ingest.py  
# または UI から /api/ingest を叩く -->

### 起動コマンド

```python
flask --app app:run run
```

 http://localhost:5000
