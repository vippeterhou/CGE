THEANO_FLAGS="device=cuda2,floatX=float32" \
    python ind_main.py \
    --dataset cora \
    --embedding_size 135 \
    --sup_iters 2000
