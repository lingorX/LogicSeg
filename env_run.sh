#!/bin/bash

# uncompress the packed enviroment into the local dir
function prepare_env() {

  tar -xf "afs/liliulei/env/loaf.tar"
  export PATH="$(pwd)/loaf/bin:$PATH"
  export LD_LIBRARY_PATH="$(pwd)/loaf/lib:$LD_LIBRARY_PATH"
}


function prepare_data() {
  START=`date +%s%N`;

#   mkdir -p data/cityscapes
#   tar -xf "afs/liliulei/datasets/seg/leftImage8bit.tar"
#   mv leftImg8bit $(pwd)/data/cityscapes
#   tar -xf "afs/liliulei/datasets/seg/gtFine.tar"
#   mv gtFine $(pwd)/data/cityscapes

  mkdir data
  tar -xf "afs/liliulei/datasets/pascal_part.tar"
  mv pascal_part data/  

#   mkdir data
#   tar -xf afs/liliulei/datasets/seg/mapillary_no_test.tar
# #   unzip afs/liliulei/datasets/seg/mapillary_val.zip
#   mv mapillary data/

#   mkdir data
#   unzip "afs/liliulei/datasets/seg/ADEChallengeData2016.zip"
#   mv ADEChallengeData2016 data/
  

  END=`date +%s%N`;
  time=$((END-START))
  time=`expr $time / 1000000000`
  echo "time for unzip dataset"
  echo $time

}


function prepare_model() {
  mkdir -p ~/.cache/torch/hub/checkpoints/
  cp afs/liliulei/pretrain/resnet101_v1c-e67eebb6.pth ~/.cache/torch/hub/checkpoints/resnet101_v1c-e67eebb6.pth
}


# unify ui
prepare_env
prepare_data
prepare_model

# ./tools/dist_train.sh local_configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_pascal_part108.py 4
./tools/dist_train.sh local_configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_pascal_part108_hiera.py 4

# ./tools/dist_test.sh local_configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_80k_pascal_part108.py iter_80000.pth 2 --eval mIoU --aug-test
# tar -cf results.tar results


