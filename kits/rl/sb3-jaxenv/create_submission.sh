#!/bin/bash
mkdir packages
JUX_PATH=$(pip show juxai_s2 | grep Location | awk '{print $2}')/jux
echo "Copying Jux" $JUX_PATH
cp -r $JUX_PATH ./packages

LUX_PATH=$(pip show luxai_s2 | grep Location | awk '{print $2}')/luxai_s2
echo "Copying Lux" $LUX_PATH
cp -r $LUX_PATH ./packages
cp -r packages/* .

rm submission.tar.gz
tar -cvzf submission.tar.gz * > /dev/null

rm -rf jux
rm -rf luxai_s2