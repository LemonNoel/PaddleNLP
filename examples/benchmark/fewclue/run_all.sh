datasets=("eprstmt" "csldcp" "tnews" "iflytek" "ocnli" "bustm" "chid" "csl" "cluewsc")

for data in ${datasets[@]}
do
    echo " "
    echo "==========" 
    echo $data
    echo "=========="
    echo " "
    bash run.sh $data 5 
done 
