<img width="1914" height="1039" alt="image" src="https://github.com/user-attachments/assets/a167eb1b-7b05-43b4-aa2e-6b92fd27fd5d" />

# Brain-GenerativeAI Interface

本プロジェクトでは、画像生成モデルに伴う高負荷な処理をクラウドにオフロードするため、AWSとローカル環境を連携させています。両環境間の通信にはFastAPIを使用しています。
LLMによるプロンプト生成、EC2へのデータ送信、Fluxモデルによる画像生成（10枚）、そしてローカル環境への返送プロセスは、**約30秒**で完了します。

## 🏛 システムアーキテクチャ

システムは以下の2つの主要コンポーネントで構成されています。

### 1. AWS環境（クラウド）
* **用途**: black-forest-labs社が提供する**FLUX.2-klein-4B**を使用した画像生成
* **メインスクリプト**: `fluxServer.py`
* **推奨環境**: 機械学習対応のEC2インスタンス（例: `g6e.2xlarge`）
    * *※検証時には `Deep Learning OSS Nvidia Driver AMI GPU Pytorch 2.10` を使用（この構成はややオーバースペックです）。*

### 2. ローカル環境
* **用途**: 多様なプロンプトの生成（Llama3.1）、EEGデータの記録（MATLAB）、システム間通信とWebUI
* **メインスクリプト**: `Ec2-client.py`, MATLABスクリプト
* **動作ツール**: Ollama (llama3.1), MATLAB, Gradio Web UI

---

## 🚀 環境構築・セットアップ手順

### AWS側のセットアップ

1. **EC2インスタンスのデプロイ**
   * 機械学習に対応したGPUインスタンスを立ち上げます。
   * サーバーにはパブリックIP経由でアクセスするため、**Elastic IP**の割り当てを推奨します。
2. **リポジトリの準備と環境構築**
   * 必要に応じてPyTorch環境をアクティベートします。`source /opt/pytorch/bin/activate`
   * 以下のコマンドで本リポジトリをクローンし、依存パッケージをインストールします。
   ```bash
   git clone [https://github.com/basic0908/nttkk-Vie2Image.git](https://github.com/basic0908/nttkk-Vie2Image.git)
   cd nttkk-Vie2Image
   pip install -r requirements.txt
   ```
   * また、HuggingFaceからモデルをダウンロードするため、`fluxServer.py`の21行目のHuggingFace Tokenを自分のトーケンに入れ替える(現在は空白)
3. サーバーの起動
  ```bash
  python fluxServer.py
  ```

### ローカル側のセットアップ
1. **Ollamaのインストールと準備**
    * Ollama公式ウェブサイト からツールをダウンロードしてインストールします。これはLLMをローカルで動かすためのオープンソースツールです。
    * ターミナルを開き、Llama 3.1モデルをダウンロード・起動します（バックグラウンドでポート 11434 を使用します）。
  ```bash
  ollama pull llama3.1
  ollama run llama3.1
  ```
2. **Matlabの起動**
    * Matlabを開き、EEGデータを記録するためのスクリプトを実行します。
3. **クライアントとUIの起動**
    * 別のターミナルを開き、`ec2_api_client.py`を実行します。
```python
python ec2_api_client.py
```

## 💻 使用方法
ローカル環境で`ec2-client.py`を実行すると、GradioによるWeb UIがローカルサーバーが(ポート7860)で立ち上がります。
1. ブラウザで`http://localhost:7860`にアクセスします。
2. Initial Concept(ベースのプロンプト)とSubject Name(被験者名)を入力します。
3. 目的に応じて以下のいずれかのアクションを実行してください:
    * **Generate 100 images** : デコーダのトレーニング用画像を生成します。
    * **Start BCI auto loop** : 実験用のBCI（Brain-Computer Interface）自動ループを開始します。
