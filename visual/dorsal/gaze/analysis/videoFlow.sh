#!/bin/bash
# 
# this is a wrapper for creating all the required files for calculating flow 
# and visualizes flow as video at the end 
#

# debug:
# show all variable values
# sh -x ./videoFlow.sh

usage()
{
    echo -e "\t Usage of the script"
    echo -e "\t "
    echo -e "\t videoFlow.sh"
    echo -e "\t Options:"
    echo -e "\t "
    echo -e "\t -flowname name of the output video"
    echo -e "\t -dir directory to save all required files"
    echo -e "\t freearg: path to input video for flow calculation"
}

COUNTER=0
DIR=`pwd`
DIR=$DIR
FREEARGCOUNTER=0
while [ "$1" != "" ]; do
    COUNTER=$((COUNTER+1))
    PARAM=`echo "$1" | awk -F= '{print $1}'`
    
    case $PARAM in
        -flowname)
            shift
            FLOWNAME=`echo "$1" | awk -F= '{print $1}'`
            ;;
        -dir)
            shift
            DIR=`echo "$1" | awk -F= '{print $1}'`
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            if [ "$FREEARGCOUNTER" -eq "0" ]; then
                FREEARG=`echo "$1" | awk -F= '{print $1}'`
                let FREEARGCOUNTER=FREEARGCOUTNER+1
            else
                echo "ERROR: unknown parameter \"$PARAM\""
                echo "ERROR: Only one argument allowed"
                usage
                exit 1
            fi  
            ;;  
    esac    
    shift
done

if [ "$COUTNER" -eq "0" ]; then
    usage
    exit 1
fi

if [[ -d $DIR ]]; then
    echo "WARNING: Directory already exists"
    echo "Might get confusing results"
else
    mkdir $DIR
fi

if [ ${DIR:$((${#DIR}-1)):1} == '/' ]; then
    DIR=${DIR:0:$((${#DIR}-1))}
fi

# get frame rate
FRAMERATE=`ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 $FREEARG | head -n 1`

# resize before separating frames and separate frames
ffmpeg -y -i $FREEARG -vf scale=1280x720 rescaled.avi
mkdir ${DIR}/frames; ffmpeg -i rescaled.avi -r $FRAMERATE $DIR/frames/%05d_frame.png


mkdir ${DIR}/edges
regex="${DIR}/frames/*.png"
matlab -nodesktop -nojvm -r "RunEdgeDetection('$regex'); exit"

# create matches files
mkdir ${DIR}/matches; files=($(ls -1 ${DIR}/frames/)); let end=${#files[@]}-2; for i in `seq 0 $end`; do out=`basename ${files[i]} .png`; out=${out}_match; echo ${files[i]}; let j=i+1; deepmatching_1.2.2_c++/deepmatching ${DIR}/frames/${files[i]} ${DIR}/frames/${files[j]} -R 2 -nt 4 -png_settings -out ${DIR}/matches/${out}; done

# create flow files
mkdir ${DIR}/flow; files=($(ls -1 ${DIR}/matches/)); let end=${#files[@]}-2
for i in `seq 0 $end`; do echo ${files[i]}; id1=`echo ${files[i]} | cut -c1-5`; let j=i+1; id2=`echo ${files[j]} | cut -c1-5`; EpicFlow_v1.00/epicflow ${DIR}/frames/${id1}_frame.png ${DIR}/frames/${id2}_frame.png ${DIR}/edges/${id1}_frame_edges ${DIR}/matches/${id1}_frame_match ${DIR}/flow/${id1}_frame_flow.flo; done

# chek if some flow files were not created. If so copy the previous frame flow to the current one
# check 1st frame
first_file="${DIR}flow/00001_frame_flow.flo"
all_files=(`ls ${DIR}flow/*_frame_flow.flo`)
if [ ! -f $first_file ]; then cp ${all_files[0]} $first_file; fi
# check the rest
for i in `seq 1 $end`; do id_bef=`echo ${files[i-1]} | cut -c1-5`; flow_file_bef="${DIR}flow/${id_bef}_frame_flow.flo"; id_cur=`echo ${files[i]} | cut -c1-5`;flow_file="${DIR}flow/${id_cur}_frame_flow.flo"; if [ ! -f $flow_file ]; then echo "Flow file does not exist $flow_file. Copying previous frame's flow file"; cp $flow_file_bef $flow_file; fi; done

# visualize flows
# black-white
for i in `ls ${DIR}/flow/*.flo`; do j=`echo $i | sed 's/\.flo//g'`; echo $j; flow-code/color_flow -bw $i ${j}.png 3; done

ffmpeg -framerate 30 -i ${DIR}/flow/%05d_frame_flow.png -c:v libx264 -r 30 -pix_fmt yuv420p ${DIR}/${FLOWNAME}"_bw.mp4"

# colored
for i in `ls ${DIR}/flow/*.flo`; do j=`echo $i | sed 's/\.flo//g'`; echo $j; flow-code/color_flow $i ${j}.png 3; done

ffmpeg -framerate 30 -i ${DIR}/flow/%05d_frame_flow.png -c:v libx264 -r 30 -pix_fmt yuv420p ${DIR}/${FLOWNAME}".mp4"


echo $FLOWNAME >> completed.txt

# clean up created files
#rm -r ${DIR}/frames ${DIR}/edges ${DIR}/matches ${DIR}/flow 
