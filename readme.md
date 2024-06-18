# Generation Networks

## Content

The exercise contains three files:

1. ```Model.py```
    - contains the model architecture including encoder and decoder
    - interface between encoder and decoder is already taken care of

2. ```train.py```
    - run this file to start the training procedure
    - contains the dataset class
    - downloads the dataset and removes all unnecessary files
    - saving of model checkpoints and data loading is already implemented

3. ```infer.py```
    - generates images from a checkpoint
    - saves them to a ```results``` sub-directory

## Tasks

Feel free to use any tricks to improve the quality of the generated images.
Minimum requirement to pass is being able to generate images from your checkpoint that at least somewhat resemble the distribution.
In the discussion session we want to show which tricks or hyperparameters have the biggest influence on the results. Please be prepared to elaborate on your thought process.

1. Build an encoder and decoder in ```Model.py```
    - define the architecture
    - implement forward pass
2. Implement a training loop in ```train.py```
3. Use ```infer.py``` to verfiy your results
4. Upload your solution as ```.zip``` file
    - please make sure **not** to zip the dataset files (dataset directory and a downloaded cache file)
    - also make sure you **do** include your trained checkpoint
5. (Optional) Did you experience mode collapse (i.e. the posterior becomes the prior)? If so, how did you resolve it?
6. (Optional) Can you make the generation process conditional, so that you can decide which of the ten classes you want to produce an image for? The class labels are thrown away during dataloading, but you can restore access to the easily.

### Potential Tips

- Normalize input images
    - do not forget to inverse the normalization when visualizing the results
- VAEs tend to be unstable
    - change the architecture if you get any ```NAN``` values in the forward pass
- if KL-loss decreases too quickly the model might collapse
    - scaling scheduler can help to resolve that issue
- images are of size ```[1, 28, 28]``` so you do not need very large networks
    - our solution trains on a laptop CPU for approximately 10 minutes
