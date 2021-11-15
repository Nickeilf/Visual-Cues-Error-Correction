GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3
TESTSET=$4

BEAM_SIZE=12


OUTDIR=results/temp

mkdir -p $OUTDIR

# CUDA_VISIBLE_DEVICES=${GPUID} nmtpy translate ${MODEL_PATH} -k ${BEAM_SIZE} -S en:${TESTSET}/test_2017_flickr.lc.norm.tok.en,feats:features/r50-avgp-224-l2/test_2017_flickr.npy -s en -o $OUTDIR/hyp.flickr \
#                                     -tid "en:Text, feats:Numpy -> en_cor:Text"
# CUDA_VISIBLE_DEVICES=${GPUID} nmtpy translate ${MODEL_PATH} -k ${BEAM_SIZE} -S en:${TESTSET}/test_2017_mscoco.lc.norm.tok.en,feats:features/r50-avgp-224-l2/test_2017_mscoco.npy -s en -o $OUTDIR/hyp.mscoco \
#                                     -tid "en:Text, feats:Numpy -> en_cor:Text"

CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_flickr.lc.norm.tok.en,image:features/butd/test_2017_flickr_obj36.npz -s en -o $OUTDIR/hyp.flickr -tid "src:Text, image:ObjectDetections -> src_cor:Text"
CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -S src:${TESTSET}/test_2017_mscoco.lc.norm.tok.en,image:features/butd/test_2017_mscoco_obj36.npz -s en -o $OUTDIR/hyp.mscoco -tid "src:Text, image:ObjectDetections -> src_cor:Text"


echo "Results on 2017 flickr set:"
# m2scorer/m2scorer $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} ${TESTSET}/test_2017_flickr.lc.norm.tok.en.m2
errant_parallel -orig ${TESTSET}/test_2017_flickr.lc.norm.tok.en -cor $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} -out $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE}.m2
errant_compare -hyp $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE}.m2 -ref ${TESTSET}/test_2017_flickr.lc.norm.tok.en.m2

python count_cor_acc.py -f $OUTDIR/hyp.flickr.en.beam${BEAM_SIZE} -gold data/00-tok/test_2017_flickr.lc.norm.tok.en -noise ${TESTSET}/test_2017_flickr.lc.norm.tok.en.marked

echo "Results on 2017 mscoco set:"
# m2scorer/m2scorer $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.m2
errant_parallel -orig ${TESTSET}/test_2017_mscoco.lc.norm.tok.en -cor $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} -out $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE}.m2
errant_compare -hyp $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE}.m2 -ref ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.m2

python count_cor_acc.py -f $OUTDIR/hyp.mscoco.en.beam${BEAM_SIZE} -gold data/00-tok/test_2017_mscoco.lc.norm.tok.en -noise ${TESTSET}/test_2017_mscoco.lc.norm.tok.en.marked
