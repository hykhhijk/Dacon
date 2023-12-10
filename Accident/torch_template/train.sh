# python train.py --epochs=200 --val_every=1 --is_split=True --loss=rmsle --batch_size=32
python train.py --epochs=200 --val_every=1 --is_split=True --loss=rmsle --batch_size=32 --loss=rmse --suffix=RMSE
python train.py --epochs=200 --val_every=1 --is_split=False --loss=rmsle --batch_size=32 --loss=rmse --suffix=RMSE


# python train.py --epochs=200 --val_every=1 --is_split=True --loss=rmsle --batch_size=128
# python train.py --epochs=200 --val_every=1 --is_split=True --loss=rmsle --batch_size=512
# python train.py --epochs=200 --val_every=1 --is_split=True --loss=rmsle --batch_size=1024


# python train.py --epochs=200 --val_every=1 --is_split=False

# python pretrain.py --epochs=200 --val_every=1 --is_split=False --batch_size=1024
