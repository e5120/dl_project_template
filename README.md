# 深層学習用プロジェクトのテンプレート

## メインで使うライブラリ
- `hydra==1.3.2`
- `wandb==0.19.4`
- `lightning==2.5.0.post0`
- `pytorch==2.3.0`

## 実行の流れ
1. パスの追加
```
> export PYTHONPATH=.
```

2. 前処理済みデータセットの作成
```
> python run/prepare_data.py 
```

3. モデルの学習
```
> python run/train.py
```

4. テストデータの推論
```
> python run/inference.py --experimental-rerun=/path/to/config.pickle
```
