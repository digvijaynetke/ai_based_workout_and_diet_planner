Here are the step-by-step instructions to install the required Python libraries using `pip`. 

### Prerequisites
- Make sure Python is installed on your system (version 3.7 or higher recommended). Verify with:
  ```bash
  python --version
  ```

### Install Required Libraries
Use the following `pip` commands to install the libraries.

#### 1. Install `Flask`
```bash
pip install flask
```

#### 2. Install `pandas`
```bash
pip install pandas
```

#### 3. Install `scikit-learn`
```bash
pip install scikit-learn
```

#### 4. Install `matplotlib`
```bash
pip install matplotlib
```

### Optional: Install All Libraries at Once
If you want to install all libraries together, use:
```bash
pip install flask pandas scikit-learn matplotlib
```

### Verify Installation
You can verify if the libraries are installed successfully:
```bash
python -c "import flask, pandas, sklearn, matplotlib; print('All libraries installed successfully')"
```

### Notes
- **Virtual Environment (Optional)**: It's a good practice to use a virtual environment to manage dependencies.
  - Create a virtual environment:
    ```bash
    python -m venv venv
    ```
  - Activate the environment:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
  - Install the libraries inside the virtual environment using the above commands.

### Troubleshooting
- If you encounter **permission errors**, add `--user`:
  ```bash
  pip install flask pandas scikit-learn matplotlib --user
  ```
- If you're using **Anaconda**, install with `conda`:
  ```bash
  conda install flask pandas scikit-learn matplotlib
  ```

Once installed, re-run your script to confirm everything works!
