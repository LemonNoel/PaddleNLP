device=$1

datasets=("eprstmt" "csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
# datasets=("eprstmt" "ocnli" "bustm" "csl")
# datasets=("iflytek" "csl")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
