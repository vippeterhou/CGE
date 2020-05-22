THEANO_FLAGS="device=cuda2,floatX=float32" \
    python trans_main.py \
    --dataset citeseer \
    --embedding_size 50 \
    --sup_iters 2000
