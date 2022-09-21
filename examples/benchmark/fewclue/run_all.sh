#device=$1

# datasets=("chid" "cluewsc")
datasets=("eprstmt" "csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl") # "cluewsc")
# datasets=("eprstmt" "ocnli" "bustm" "csl")

device=0
for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device > $data.log 2>&1&
    device=`expr $device + 1`
done 
