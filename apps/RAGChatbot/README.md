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


## セットアップ

```sh
pip install -r requirements.txt
```

`.env` を作成して必要な値を設定してください。

```env
OPENAI_API_KEY=your_key
SERP_API_KEY=your_key  # Web/Hybridモードで必須
DATABASE_URL=postgresql+psycopg2://appuser:password@localhost:15432/appdb
SCOPE_KEYWORDS=補助金,助成金,給付金,支援制度,支援金,助成制度
```

`DATABASE_URL` を使わない場合は、以下でも接続できます。

```env
POSTGRES_USER=appuser
POSTGRES_PASSWORD=password
POSTGRES_HOST=localhost
POSTGRES_PORT=15432
POSTGRES_DB=appdb
```

PDFを入れるフォルダは `data/pdf` です。

任意の制限（必要な場合のみ）:

```env
INGEST_MAX_FILES=200
INGEST_MAX_BYTES=20000000
```

### 事前インデックス
<!-- python scripts/ingest.py  
# または UI から /api/ingest を叩く -->

### 起動コマンド

```sh
flask --app app:create_app run
```

または

```sh
python run.py
```

 http://localhost:5000

### DB（任意）

PostgreSQL を使う場合は `docker-compose.yml` を起動します。

```sh
docker compose up -d
```
