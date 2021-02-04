python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-4sample-50prob --sensitive --batch_size=96 --num_aug=4

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-3sample-50prob --sensitive --batch_size=96 --num_aug=3

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-2sample-50prob --sensitive --batch_size=96 --num_aug=2

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-1sample-50prob --sensitive --batch_size=96 --num_aug=1
