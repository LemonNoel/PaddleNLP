device=$1

datasets=("tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
# datasets=("eprstmt" "ocnli" "bustm" "csl")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
