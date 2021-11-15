GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3
TESTSET=$4

BEAM_SIZE=12


OUTDIR=results/temp

mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_flickr.lc.norm.tok.en,image:features/butd/test_2017_flickr_obj36.npz -s src -o $OUTDIR/hyp.flickr
CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_mscoco.lc.norm.tok.en,image:features/butd/test_2017_mscoco_obj36.npz -s src -o $OUTDIR/hyp.mscoco

echo "Results on 2017 flickr set:"
nmtpy-coco-metrics $OUTDIR/hyp.flickr.src.beam${BEAM_SIZE} -l ${TRG_LANG} -r data/00-tok/test_2017_flickr.lc.norm.tok.${TRG_LANG}
echo "Results on 2017 mscoco set:"
nmtpy-coco-metrics $OUTDIR/hyp.mscoco.src.beam${BEAM_SIZE} -l ${TRG_LANG} -r data/00-tok/test_2017_mscoco.lc.norm.tok.${TRG_LANG}