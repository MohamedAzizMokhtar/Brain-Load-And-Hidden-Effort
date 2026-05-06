## Usage

### Option 1: Run the complete pipeline in Kaggle (Recommended)

Since fMRI processing requires significant computational resources, the easiest way to run this project is on Kaggle:

1. **Upload the notebooks to Kaggle**
   - Go to [Kaggle.com](https://www.kaggle.com)
   - Create a new notebook
   - Upload `notebooks/data-preprocessing.ipynb`
   - Upload `notebooks/modeling-anomaly-detection.ipynb`

2. **Set up GPU accelerator**
   - Click "Settings" ? "Accelerator" ? "GPU T4 x2"

3. **Add your dataset**
   - Click "Add Data" ? Upload your fMRI .nii files

4. **Run sequentially**
   - First run `data-preprocessing.ipynb` (generates preprocessed clips)
   - Then run `modeling-anomaly-detection.ipynb` (trains models and evaluates)

### Option 2: Local execution (CPU only - slower)

If you have Python installed locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run preprocessing notebook
jupyter notebook notebooks/data-preprocessing.ipynb

# Then run modeling notebook
jupyter notebook notebooks/modeling-anomaly-detection.ipynb