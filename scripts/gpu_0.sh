cd ../


python run.py model=rgcl \
  data.dataset=Amazon_Musical_Instruments_14 \
  evaluation.eval_only=true \
  evaluation.checkpoint_path=/home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/rgcl_Amazon_Musical_Instruments_14_64_20260603_164647/rgcl_best.pt \
  evaluation.sentiment_subset=sentiment_pos

python run.py model=deepconn \
  data.dataset=Amazon_Musical_Instruments_14 \
  evaluation.eval_only=true \
  evaluation.checkpoint_path=/home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/deepconn_Amazon_Musical_Instruments_14_42_20260603_174332/deepconn_best.pt \
  evaluation.sentiment_subset=sentiment_pos

python run.py model=narre \
  data.dataset=Amazon_Musical_Instruments \
  evaluation.eval_only=true \
  evaluation.checkpoint_path=/home/infolab/mnt/mingyu/review_rec/review_reproducibility/outputs/neumf_Amazon_Musical_Instruments_42_20260604_115308/neumf_best.pt \
  evaluation.sentiment_subset=sentiment_pos