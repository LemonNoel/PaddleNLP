device=$1
datasets=("ocnli" "bustm" "cluewsc")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
