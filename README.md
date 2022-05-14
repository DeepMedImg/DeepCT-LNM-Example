# DeepCT-LNM-Example

This is an example of running inference with the trained DeepCT-LNM prediction model. The model takes as inputs the CT image (arterial and portal phases) and corresponding PDAC and lymph node masks, and outputs the probability of lymph node metastasis status. An illustrative example of the input data is provided in the *test_example* folder. This code is developed based on [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.

## Installation
This code depends on [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Below are quick steps for installation. Please refer to (https://github.com/MIC-DKFZ/nnUNet#installation) for more detailed instruction.

1) Install [PyTorch](https://pytorch.org/get-started/locally/) 

  ```pip install torch torchvision```
  
2) Install [nnUNet](https://github.com/MIC-DKFZ/nnUNet)

  ```pip install nnunet```
  
## Usage
The DeepCT-LNM prediction model is avaiable for research-use only. COMMERCIAL USE IS PROHIBITED for the time being. 

### set environment variables

Set *checkpoints path* to RESULTS_FOLDER

  ```export RESULTS_FOLDER="checkpoints path"```

### run inference with the trained model
  ```bash run_inference.sh test_data_dir```
  
 

