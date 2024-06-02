# LiverSurvival

This repository contains the necessary scripts and instructions to run a complete pipeline including data preprocessing, training, and testing phases using a CNN model with DenseNet121 as the backbone.

## Data Preprocessing

First, prepare your data by running the preprocessing script. Ensure that your raw data is located in the correct directory or modify the script accordingly.

```bash
python data/preprocessing.py
```

## Training the Model

Use the following command to train the network. The script uses DenseNet121 as the backbone, and various parameters can be adjusted based on your hardware and dataset specifics.

```bash
python train_cnn.py \
    --backbone 'densenet121' \
    --random_seed 10 \
    --lr 5E-03 \
    --nb_epoch 70 \
    --batch_size 16 \
    --n_cpu 16 \
    --output_folder 'OUTPUT_PATH' \
    --gpus 0 \
    --gpu_1 0 \
    --gpu_2 1 \
    --fold 0 \
    --norm 'bn' \
    --br 6 \
    --cat 'ct'  
```

## Testing the Model

After training, test the model to evaluate its performance using the following command. Be sure to replace the `output_folder` with the path where your trained model is stored.

```bash
python test.py \
    --backbone 'densenet121' \
    --random_seed 10 \
    --n_cpu 16 \
    --output_folder 'OUTPUT_PATH' \
    --gpus 0 \
    --fold 0 \
    --norm 'bn' \
    --cat 'ct'
```

For further details on the scripts and parameters, please refer to the documentation in each script's header or the additional documentation provided.

## Contact

For any questions or issues, please open an issue on this GitHub repository or contact the maintainers directly through the provided contact links.
