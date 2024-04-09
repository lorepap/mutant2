filename=$1
out_dir=$2
# log=log/mab/mahimahi/$timestamp
# filename=$log/uplink.log

echo "Doing Analysis ..."
echo "------------------------------"
out="$filename.tr"
# mkdir -p log
echo $out_dir >> $out
./mm-thr 500 $filename 1>$filename.svg 2>res_tmp
cat res_tmp >> $out

./mm-del-file $filename 2>res_tmp2
# ./plot-del.sh log/down-$log.dat
echo "------------------------------" >> $out
cat res_tmp
echo "------------------------------"

rm res_tmp res_tmp2

echo "Finished."
