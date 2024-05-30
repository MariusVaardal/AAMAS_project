# AAMAS_project

This project requires a Python 3.10 environment. Follow the steps below to run the .exe demo-script!

## Setup Instructions

### 0. Clone the repo to your computer

Clone the github repository to somewhere on your computer. This can be done with the following command 

```bash 
git clone https://github.com/MariusVaardal/AAMAS_project.git
```

### 1. Create a Python 3.10 Virtual Environment

First, navigate to your project's root directory (same directory as README.md file) where you want to create the virtual environment.

```bash
cd /path/to/your/project
```

Create the virtual environment using Python 3.10:

```bash
python3.10 -m venv venv
```

This will create a directory named `venv` in your project directory.

### 2. Activate the Virtual Environment

#### On Windows

```bash
.\venv\Scripts\activate
```

#### On macOS and Linux

```bash
source venv/bin/activate
```

After activation, you should see the virtual environment's name (`venv`) in your terminal prompt, indicating that the environment is active. NB: If you are using Windows Powershell and get an "unauthorized access"-error try changing to cmd by running the following command before activating the venv:

```bash
cmd
```


<!-- ### 3. Install Packages from `requirements.txt`

Ensure you are in the project's root directory where the `requirements.txt` file is located. Then run:

```bash
pip install -r requirements.txt
```

This will install all the packages listed in the `requirements.txt` file into your virtual environment. -->

### 3. Download and activate git lfs 

#### On Windows

Make sure you have git lfs (large file storage) installed by running the command: 

```bash
git lfs version
```

If it is not installed, it can be installed here: [git installation](https://gitforwindows.org/)

#### On macOS and Linux

```bash
brew update
brew install git-lfs
```

### 4. Initialize git lfs in the root directory of your cloned repo.

Do this by running the command:

#### On Windows

```bash
git lfs install
```

#### On macOS and Linux

```bash
git lfs install
```


### 5. Download the files that were too large for github

Do this by running the following commands

#### On Windows

```bash
git lfs pull
```

#### On macOS and Linux

```bash
git lfs pull
```

### 6. Unzip the downloaded .zip files

Unzip the build.zip and dist.zip files into the same directory (root directory) with the same name. You can do this manually or using a command, but the important thing is that you end up with a directory named "build" and a directory named "dist" in the project's root directory (same directory as the README.md file). Note that the dist.zip is large and will take some time to extract. 

On windows we recommend unzipping manually, but in Linux, you can use the commands:

```bash
unzip build.zip
unzip dist.zip
```

### 7. If you are using Linux or MacOS:

Make sure you have permission to run the exe file by running the command: 

```bash
chmod +x ./dist/demo/demo.exe
```

### 8. Run the executable

Run the executable by using the command: 

### On MacOS and Linux

```bash
./dist/demo/demo.exe
```

### On Windows

```bash
.\dist\demo\demo.exe
```

Note that if you extracted the files by making another folder called "dist", you will have to run:

### On Windows

```bash
.\dist\dist\demo\demo.exe
```

### On MacOS and Linux

```bash
./dist/dist/demo/demo.exe
```

Also note that it might be a little slow at first (especially in Linux, we have experienced), in order to install all the required packages. If it takes too long, just kill the terminal and try the last command again. If, for some reason, you get an error message complaining about some packages not being installed, simply install the requirements by running the command:

```bash 
pip install -r requirements.txt
```

and try again. 

## 9. Watch our demo!

Now you can see our agents in action in addition to some plots that will show you the agent's performances in the form of graphs. Note that the demo is interactive and you have to click "OK" on the message boxes that appear on the screen in order for the program to continue. You also have to close the plots once you are done looking at them so that the next plots are generated and shown. 

## 10. Deactivating the Virtual Environment

Once you are done working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```

