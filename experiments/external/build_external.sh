#!/usr/bin/env bash
# Builds the original authors' implementations used in the comparison
# with original code (Section "Comparison with the original implementations").
#   Exponion, Yinyang, Hamerly: eakmeans by J. Newling (Idiap), BSD-3
#   Ball k-means: C++ release by S. Xia et al.
# Requirements: g++ (C++11), git, Eigen 3 (e.g. apt install libeigen3-dev).
set -e
cd "$(dirname "$0")"
EAK_COMMIT=185f9349d99d8d64fde18ae4c219628405eefd98
BALL_COMMIT=87cddcca2dee225a7863b492ceda082a92a07d67
[ -d eakmeans ] || git clone https://github.com/idiap/eakmeans.git
(cd eakmeans && git checkout -q $EAK_COMMIT && \
  sed -i 's/^USEBLAS = YES/USEBLAS = NO/' Makefile && make main)
[ -d ball-k-means ] || git clone https://github.com/syxiaa/ball-k-means.git
(cd ball-k-means && git checkout -q $BALL_COMMIT)
# The released file uses float; switch to double so that results can be
# compared exactly with Lloyd's algorithm, and replace the hard-coded
# main() by a command-line entry point (data, centroids, ring flag, labels).
python3 patch_ball.py ball-k-means/C++Version/ball_kmeans++_xd.cpp ballkm.cpp
EIGEN=${EIGEN_INCLUDE:-/usr/include/eigen3}
g++ -O3 -march=native -std=c++11 -I"$EIGEN" ballkm.cpp -o ballkm
echo "built: eakmeans/bin/blaslesskmeans and ./ballkm"
