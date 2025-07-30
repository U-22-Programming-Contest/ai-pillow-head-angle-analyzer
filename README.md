# ai-pillow-head-angle-analyzer

本リポジトリは、AI枕プロジェクトの一環として、**寝ている人の頭部と水平面とのなす角度**をカメラ映像から推定するシステムです。  
この推定結果は、今後のAI枕の**自動高さ・形状調整の制御パラメータ**としての活用を目的としています。

---

## 仕様

### 🎯 目的

- 枕に頭部を載せた被験者の姿勢（特に**頭部と水平面のなす角度**）を画像・映像から推定し、適切な枕の高さ・形状調整に活用する。

---

### 📷 入力

- カメラで撮影した寝姿画像または動画（例：`.jpg`, `.mp4` など）
- 想定カメラ位置：**枕の側面からの横向き視点**

---

### 🧠 出力

- 頭部と水平面のなす角度（単位：度）
- 姿勢可視化画像（鼻・肩・腰を結ぶ線、角度ラベル付き）

#### 出力イメージ

<img src="image/image_output.jpg" width="600">

---

### ⚙️ 処理の流れ

1. 寝姿画像または動画の読み込み
2. MediaPipe Pose によるキーポイント（鼻・肩・腰）検出
3. ベクトル演算による頭部の傾き推定
4. 水平方向とのなす角度を算出
5. 推定角度と可視化画像を出力

---

## 🔧 実行環境

- Python 3.11.9
- venv による仮想環境での実行

---

## 🚀 実行方法

### 1. 仮想環境の作成と有効化

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 必要ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. メインスクリプトの実行

```bash
python main.py
```

## 📁 ディレクトリ構成

```bash
ai-pillow-head-angle-analyzer/
├── image/
│   └── image_output.jpg       # 出力例画像
├── datas/
│   └── mv2.mov                # 入力動画
│   └── 1.jpg                  # 入力画像
├── main.py                    # メインスクリプト
├── demo.mp4                   # デモ動画
├── requirements.txt
└── README.md
```

## 開発者
佐藤　光河