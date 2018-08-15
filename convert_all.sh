#!/bin/bash


if [ $# -lt 1 ]; then
    echo "Usage: ./convert_all.sh <stage> [<limit>]"
    exit 1
fi

stage=$1

limit=0
if [ $# -gt 1 ]; then
    limit=$2
fi

num_chunks=12

seq 1 $num_chunks | xargs -P $num_chunks -I % python \
    convert_to_tfrecord.py --stage $stage --chunk % \
    --gzip --limit $limit --num_chunks $num_chunks

