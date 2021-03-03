dev=1
BASE=$(dirname $(pwd))
DATA_BASE=$(dirname $(dirname $(pwd)))
RUN_PATH=$BASE/fairseq_cli
export PYTHONPATH=$BASE
mode=$1
arch=realformer_wmt_en_de

if [ "$mode" == "prepare" ]
then
echo "not implemented!!"
# Preprocess/binarize the data
# TEXT=$BASE/examples/translation/iwslt14.tokenized.de-en
# python $RUN_PATH/preprocess.py --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en \
#     --workers 20

elif [ "$mode" == "train" ]
then
echo "Start training..."
MODEL_PATH=checkpoints/${arch}
mkdir -p $MODEL_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u $RUN_PATH/train.py data-bin/wmt14_en_de \
	--arch $arch --share-all-embeddings \
	--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
	--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	--lr 0.0007 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
	--max-tokens 4096 --max-update 250000 \
	--update-freq 2 --no-progress-bar --log-format json --log-interval 50 \
	--save-interval-updates  10000 --keep-interval-updates 20 \
	--save-dir $MODEL_PATH \
	--fp16 \
	--no-epoch-checkpoints \
    --save-dir $MODEL_PATH > $MODEL_PATH/train.log 2>&1 &

elif [ "$mode" == "test" ]
then
    MODEL_PATH=checkpoints/${arch}
	PYTHONIOENCODING=utf-8 python -u $RUN_PATH/generate.py data-bin/wmt14_en_de \
    --path $MODEL_PATH/checkpoint_best.pt \
    --batch-size 32 --beam 5 --remove-bpe --lenpen 0.6 --fp16 2>&1 | tee $MODEL_PATH/test.log
fi
