#!/bin/bash

DIRECTORY=$(cd `dirname $0` && pwd)
HEADER="${DIRECTORY}/all_headers.h"
yourfilenames=`ls $DIRECTORY/protocol/*.h`
counter=0
protocols=()

# Empty file
> $HEADER

# Add includes
echo "#ifndef __ALL_MIMIC_HEADERS__" >> $HEADER
echo "#define __ALL_MIMIC_HEADERS__ 1" >> $HEADER
echo "" >> $HEADER
echo "#include <net/tcp.h>" >> $HEADER
echo "#include \"mimic.h\"" >> $HEADER

for file in $yourfilenames
do

    IFS='/' read -r -a pathItems <<< $file
    lastItem="${pathItems[-1]}"

    echo "#include \"protocol/$lastItem\"" >> $HEADER
    echo $file
    counter=$(( $counter + 1 ))

    IFS='.' read -r -a nameItems <<< $lastItem
    protocols+=(${nameItems[0]})

done

echo "" >> $HEADER
echo "#define ARM_COUNT $counter" >> $HEADER


# Generate init message
echo "" >> $HEADER

counter=0
msg=''
seperator=';'
semi=':'
for ps in ${protocols[@]}
do
    line=${counter}${semi}${ps}${seperator}
    msg=${msg}${line}
    counter=$(( $counter + 1 ))
done
echo "#define INIT_MSG \"$msg\"" >> $HEADER

echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER

# Generate init function

echo "static void doInit(struct sock *sk, struct arm *ca)" >> $HEADER
echo "{" >> $HEADER

for ps in ${protocols[@]}
do
    echo "" >> $HEADER
    echo "      ${ps}_init(sk, ca);" >> $HEADER

done

echo "" >> $HEADER
echo "}" >> $HEADER


echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER


# Generate on change congestion window size function
counter=0
echo "static __u32 changeCongestionWindowSize(struct sock *sk, __u32 ack, __u32 acked, __u32 protocolId, struct arm *arm)" >> $HEADER
echo "{" >> $HEADER
echo "      switch (protocolId)" >> $HEADER
echo "      {" >> $HEADER
echo "" >> $HEADER

for ps in ${protocols[@]}
do
    echo "          case ${counter}:" >> $HEADER
    echo "              return ${ps}_cong_avoid(sk, ack, acked, arm);" >> $HEADER
    echo "" >> $HEADER
    counter=$(( $counter + 1 ))

done

echo "          default:" >> $HEADER
echo "              return cubic_cong_avoid(sk, ack, acked, arm);" >> $HEADER

echo "" >> $HEADER
echo "     }" >> $HEADER
echo "}" >> $HEADER


echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER
echo "" >> $HEADER


# Generate on change packet acked function
counter=0
echo "static void doPacketsAcked(struct sock *sk, const struct ack_sample *sample, __u32 protocolId, struct arm *arm)" >> $HEADER
echo "{" >> $HEADER
echo "      switch (protocolId)" >> $HEADER
echo "      {" >> $HEADER
echo "" >> $HEADER

for ps in ${protocols[@]}
do
    echo "          case ${counter}:" >> $HEADER
    echo "              return ${ps}_acked(sk, sample, arm);" >> $HEADER
    echo "" >> $HEADER
    counter=$(( $counter + 1 ))

done

echo "          default:" >> $HEADER
echo "              return cubic_acked(sk, sample, arm);" >> $HEADER

echo "" >> $HEADER
echo "     }" >> $HEADER
echo "}" >> $HEADER

echo "" >> $HEADER
echo "#endif" >> $HEADER