#!/bin/bash

# Clear output file
> out.csv

while read f; do

    # read in line to array, comma delimited
    arrIn=(${f//,/ })

    printf -v plate "%04d" ${arrIn[0]}
    printf -v mjd "%05d" ${arrIn[1]}
    printf -v fiber "%03d" ${arrIn[2]}

    # Example: http://das.sdss.org/raw/spectro/1d_26/0308/1d/spSpec-51662-0308-640.fit
    out="http://das.sdss.org/raw/spectro/1d_26/$plate/1d/spSpec-"
    out="$out$mjd-$plate-$fiber.fit"

    echo $out >> out.csv

done < <(tail -n+2 result.csv)

