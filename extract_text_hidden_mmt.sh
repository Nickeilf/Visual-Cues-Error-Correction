GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3
TESTSET=$4

OUTDIR=results/semantic

CUDA_VISIBLE_DEVICES=${GPUID} pysimt simtest ${MODEL_PATH} -b 1 -k 1 -S src:${TESTSET}/test_2017_flickr.lc.norm.tok.en,image:features/butd/test_2017_flickr_obj36_shuf.npz -s en -o $OUTDIR/mmt_cor.flickr.noisy