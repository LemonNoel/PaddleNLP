device=$1

# datasets=("chid" "cluewsc")
datasets=("eprstmt" "csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
# datasets=("csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
# datasets=("ocnli" "bustm" "iflytek")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
