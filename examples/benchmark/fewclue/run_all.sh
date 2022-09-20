device=$1

datasets=("csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")
# DONE: "eprstmt"
# datasets=("csldcp" "iflytek" "chid" "csl" "cmnli")
# datasets=("ocnli" "bustm")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data $device
done 
