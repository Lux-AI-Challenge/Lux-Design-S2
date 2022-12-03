#!/bin/sh
echo 'Generate source list'
find . -name '*.java' > sources.txt
echo 'Compile source'
javac -classpath ./lib/jackson-core-2.13.4.jar:./lib/jackson-databind-2.13.4.jar:./lib/jackson-annotations-2.13.4.jar -d ./build @sources.txt
echo 'Add external libs'
unzip -o './lib/*.jar' -d ./build
echo 'Move to build folder'
cd ./build
echo 'Create fat-jar'
jar -cfe JavaBot.jar com.luxai.Bot com/
echo 'Move file'
mv -f JavaBot.jar ../JavaBot.jar
echo 'end'