GPUID=$1
TRG_LANG=$2
MODEL_PATH=$3



OUTDIR=results/temp

mkdir -p $OUTDIR

# CUDA_VISIBLE_DEVICES=${GPUID} pysimt attentionvisual ${MODEL_PATH} -b 1 -k 1 -S src:data/attention_example.en,image:features/butd/attention_example.npz -s en -o $OUTDIR/attention

CUDA_VISIBLE_DEVICES=${GPUID} pysimt attentionvisual ${MODEL_PATH} -b 1 -k 1 -S src:data/attention_example.en,image:features/butd/attention_example2.npz -s en -o $OUTDIR/attention -tid "src:Text, image:ObjectDetections -> src_cor:Text"

