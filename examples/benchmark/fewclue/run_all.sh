device=$1

#datasets=("eprstmt" "csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
datasets=("eprstmt" "csldcp" "tnews")
#datasets=("iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
