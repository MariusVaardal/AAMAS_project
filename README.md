# AAMAS_project

We were unfortunately unable to create a single executable file from the project as we ran into bugs.

However, we still provide a guide to fairily easily run the project.

---

This project requires a Python 3.10 environment. Follow the steps below to create and activate a virtual environment, and then install the necessary packages from the `requirements.txt` file.

## Setup Instructions

### 1. Create a Python 3.10 Virtual Environment

First, navigate to your project's root directory where you want to create the virtual environment.

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

After activation, you should see the virtual environment's name (`venv`) in your terminal prompt, indicating that the environment is active.

### 3. Install Packages from `requirements.txt`

Ensure you are in the project's root directory where the `requirements.txt` file is located. Then run:

```bash
pip install -r requirements.txt
```

This will install all the packages listed in the `requirements.txt` file into your virtual environment.


### 4. Important: Change project folder

Navigate to `utils.py` and change the variable `PROJECT_PATH` to be equal to the path where the project is located. Save the file.

### 5. Run the project

This command will run the project. The videos will be displayed followed by a reproduction of the plots from the report.

```bash
python simpleTagCoordination.py
```


## Deactivating the Virtual Environment

Once you are done working in the virtual environment, you can deactivate it by running:

```bash
deactivate
```

