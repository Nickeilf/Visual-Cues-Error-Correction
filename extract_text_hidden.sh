GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3
TESTSET=$4

OUTDIR=results/semantic

CUDA_VISIBLE_DEVICES=${GPUID} pysimt simtest ${MODEL_PATH} -b 1 -k 1 -S src:${TESTSET}/test_2017_flickr.lc.norm.tok.en -s en -o $OUTDIR/nmt_cor.flickr.noisy