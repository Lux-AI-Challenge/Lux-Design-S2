# LUX-AI 2022 JAVA BOT
## LOCAL RUN
1. Create fat-jar by maven or javac (see below)
2. Move jar to subfolder localrun
3. Rename to JavaBot.jar
4. Run with javamediator.py
```
luxai2022 javamediator.py javamediator.py -o replay.json
```
#### Fat-jar by maven
```bash
mvn clean compile assembly:single
```
#### Fat-jar by javac
1. Move to project folder. ${project.basedir} is your project folder that contain src and lib folder
    ```
    cd ${project.basedir}
    ```
2. Generate source list
    ```
    dir /s /B .\src\main\java\*.java > sources.txt
    ```
3. Compile source
    ```
    javac -classpath ".\lib\jackson-core-2.13.4.jar;.\lib\jackson-databind-2.13.4.jar;.\lib\jackson-annotations-2.13.4.jar" -d ".\build" @sources.txt
    ```
4. Add external libs
    ```
    unzip -o ".\lib\*.jar" -d .\build
    ```
5. Move to build folder
    ```
    cd .\build
    ```
6. Create fat-jar
    ```
    jar -cfe JavaBot.jar com.luxai.Bot com/
    ```
	