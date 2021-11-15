GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3


BEAM_SIZE=$4

data="$(cut -d '/' -f2 <<<"${MODEL_PATH}")"
model="$(cut -d '/' -f3 <<<"${MODEL_PATH}")"

OUTDIR=results/$data/$model/

mkdir -p $OUTDIR

CUDA_VISIBLE_DEVICES=${GPUID} pysimt translate ${MODEL_PATH} -f bs -k ${BEAM_SIZE} -s test_2017_flickr,test_2017_mscoco -o $OUTDIR/hyp

echo "Results on 2017 flickr set:"
nmtpy-coco-metrics $OUTDIR/hyp.test_2017_flickr.beam${BEAM_SIZE} -l ${TRG_LANG} -r data/00-tok/test_2017_flickr.lc.norm.tok.${TRG_LANG}
echo "Results on 2017 mscoco set:"
nmtpy-coco-metrics $OUTDIR/hyp.test_2017_mscoco.beam${BEAM_SIZE} -l ${TRG_LANG} -r data/00-tok/test_2017_mscoco.lc.norm.tok.${TRG_LANG}
