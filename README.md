# LiverSurvival

This repository contains the necessary scripts and instructions to run a complete pipeline including data preprocessing, training, and testing phases using a CNN model with DenseNet121 as the backbone.

## Liver Region Segmentation
To extract the liver mask from CT scans, use TotalSegmentator. Install the necessary libraries with the following command:

```bash
pip install TotalSegmentator
```

Then, to generate a liver mask file (liver.nii.gz) from a CT scan, use:

```bash
TotalSegmentator -i CT_FILE_PATH(.nii.gz) -o OUTPUT_FOLDER_PATH --task total
```

Find TotalSegmentator on [GitHub](https://github.com/wasserth/TotalSegmentator)

## Data Structure

Organize your data in the following structure:

```
- Volume
    - 001_123123.nii.gz
    - 002_124124.nii.gz
    - ...
- Mask
    - 001_123123.nii.gz
    - 002_124124.nii.gz
    - ...
- Preprocessed
    - 001_123123.npy
    - 002_124124.npy
    - ...
```

## Data Preprocessing

Prepare your data by running the preprocessing script. Ensure that your raw data is located in the correct directory or modify the script accordingly.

```bash
python data/preprocessing.py -f FOLD_PATH(.json) -e EXCEL_PATH(.xlsx)
```

- `FOLD_PATH` refers to the JSON file containing patient numbers for each fold (train/validation/test).
- `EXCEL_PATH` refers to the XLSX file containing patient information.

For reference, example files randomly generated are available in the `example` folder.

## Training the Model

Use the following command to train the network. 
The script uses DenseNet121 as the backbone, and various parameters can be adjusted based on your hardware and dataset specifics.

```bash
python train_cnn.py \
    --backbone 'densenet121' \
    --lr 5E-03 \
    --nb_epoch 70 \
    --batch_size 16 \
    --gpus 0 \
    --fold 0 \
    --br 6 \
    --cat 'ct'  \
    --output_folder 'OUTPUT_PATH' \
    --fold_path 'FOLD_PATH' \
    --excel_path 'EXCEL_PATH' 
```

- `OUTPUT_PATH` is the directory where the model will be saved.

## Testing the Model

After training, test the model to evaluate its performance using the following command. 
Results for each metric can be found in `result.json` within the `OUTPUT_PATH`.

```bash
python test.py \
    --backbone 'densenet121' \
    --gpus 0 \
    --fold 0 \
    --cat 'ct' \
    --output_folder 'OUTPUT_PATH' \
    --fold_path 'FOLD_PATH' \
    --excel_path 'EXCEL_PATH' 
```

## Contact

For any questions or issues, please open an issue on this GitHub repository
