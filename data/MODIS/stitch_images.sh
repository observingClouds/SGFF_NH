#!/bin/bash


#split_chars() { sed $'s/./&\\\n/g' <<< "$1";  }
#r=`comm --nocheck-order -12 <(split_chars "filename_1_22") <(split_chars "filename_1_23")`
#echo $r | tr -d " "

OUTDIR=TropicalBelt

cd $1 # Assuming all given directories have the same parent directory
cd ..

mkdir $OUTDIR

for file in `ls $1`;
do
	echo $file
	pattern=`echo ${file:0:40}`
	files=$(printf "%s/$pattern* " "$@")
	output=$OUTDIR/${pattern}_-180-180_-15-15.jpeg
	convert $files +append $output
done

