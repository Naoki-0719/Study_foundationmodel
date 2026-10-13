# UAD評価コード：NO GPT版（公開モデルの取得は許可）

## ポリシー
- **禁止**：GPT/OpenAI等のクラウドLLM・外部推論APIへ画像、GT、プロンプト、推論結果を送信すること。
- **許可**：GitHub/Hugging Face/PyTorch Hubから公開モデル本体・学習済み重み・依存パッケージを取得する通信。
- UADテキスト埋め込みは `embeddings_st` のみ許可します。`oai_emb.yaml`や`eval_agd.yaml`などOpenAI埋め込み構成はコードで拒否します。
- 本コードは画像・GTをローカルの公式UAD推論器へ渡します。**サードパーティ依存パッケージや公式UAD自体の全通信を監査・遮断する仕組みではありません**。機密データ投入前に組織の規則に従いコード・通信先を監査してください。

## 使用方法
公式UADを `unsup-affordance/` 以下に準備し、`checkpoints/st_emb.pth` を取得します。既存SAM3と同じ `DATA/MeasureData`, `DATA/Annotation`, `prompts_affordance.tsv`, `affordance_colors.tsv` を使用します。

```bash
bash run_all_affordance_UAD.sh --check_only
bash run_all_affordance_UAD.sh --save_scores
```

モデルのDINOv2やSentence Transformers重みがローカルキャッシュに無ければ公式ライブラリがネットから取得することがあります。API呼び出し型の埋め込みは事前に禁止します。評価画像とGTを外部へ送信する処理は本評価コードにはありません。

**注意**：実際のUAD学習済み重みによる推論はこの環境では未検証です。実験投入前に機密情報のないダミー画像で実行検証してください。
