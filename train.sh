#!/bin/bash

cat train_args.txt | while read line; do
    row=(`echo ${line}`)
    args="--model ${row[0]}"
    args="${args} --classes ${row[1]}"
    args="${args} --optimizer ${row[2]}"
    args="${args} --targetsize-width ${row[3]}"
    args="${args} --targetsize-height ${row[4]}"
    args="${args} --batchsize ${row[5]}"
    args="${args} --epochs ${row[6]}"
    args="${args} --rotation-range ${row[7]}"
    args="${args} --width-shift-range ${row[8]}"
    args="${args} --height-shift-range ${row[9]}"
    args="${args} --shear-range ${row[10]}"
    args="${args} --zoom-range ${row[11]}"
    if [[ `echo ${row[12]} | tr [:lower:] [:upper:]` == "TRUE" ]]; then
        args="${args} --horizontal-flip"
    fi

    if [[ `echo ${row[13]} | tr [:lower:] [:upper:]` == "TRUE" ]]; then
        args="${args} --vertical-flip"
    fi
   echo ${args}
   python train_by_json.py ${args}
done