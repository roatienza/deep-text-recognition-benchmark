python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-8sample-50prob --sensitive --batch_size=96 --num_aug=8

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-7sample-50prob --sensitive --batch_size=96 --num_aug=7

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-6sample-50prob --sensitive --batch_size=96 --num_aug=6

python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn --manualSeed=$RANDOM --exp_name=rare-5sample-50prob --sensitive --batch_size=96 --num_aug=5
