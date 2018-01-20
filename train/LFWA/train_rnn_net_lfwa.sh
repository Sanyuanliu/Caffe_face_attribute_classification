./build/tools/caffe train \
    --solver=train/LFWA/solver_lfwa.prototxt \
    2>&1 | tee train/LFWA/attribute.log
#    --weights=/home/sanyuan/CaffeProject/FA_v1-attention/models/Attribute/lfwa/step1/lfwa_iter_10000.caffemodel \
