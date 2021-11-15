GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3
TESTSET=$4

BEAM_SIZE=12


OUTDIR=results/temp

mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_flickr.lc.norm.tok.en -s en -o $OUTDIR/hyp.flickr \
                                    -tid "src:Text -> src_cor:Text"
CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_mscoco.lc.norm.tok.en -s en -o $OUTDIR/hyp.mscoco \
                                    -tid "src:Text -> src_cor:Text"


echo "Results on 2017 flickr set:"
# python m2scorer/scripts/m2scorer.py $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} ${TESTSET}/test_2017_flickr.lc.norm.tok.en.m2
errant_parallel -orig ${TESTSET}/test_2017_flickr.lc.norm.tok.en -cor $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} -out $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE}.m2
errant_compare -hyp $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE}.m2 -ref ${TESTSET}/test_2017_flickr.lc.norm.tok.en.m2
python count_cor_acc.py -f $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} -gold data/00-tok/test_2017_flickr.lc.norm.tok.en -noise ${TESTSET}/test_2017_flickr.lc.norm.tok.en.marked

echo "Results on 2017 mscoco set:"
# python m2scorer/scripts/m2scorer.py $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.m2
errant_parallel -orig ${TESTSET}/test_2017_mscoco.lc.norm.tok.en -cor $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} -out $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE}.m2
errant_compare -hyp $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE}.m2 -ref ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.m2
python count_cor_acc.py -f $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} -gold data/00-tok/test_2017_mscoco.lc.norm.tok.en -noise ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.marked
