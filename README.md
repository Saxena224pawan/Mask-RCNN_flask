# Mask-RCNN_flask
## About
This is maskRCNN implementation in flask.  
## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python app.py
    ``` 
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from (https://github.com/matterport/Mask_RCNN/releases). and save this model as your_model.h5
5. You also need to pycocotools from the links given below:
    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
    
## Screenshot:

![Segmentation Sample](cat_screenshot.png)

## Credits:
Some of the code has been taken from [Matterport's MaskRCNN](https://github.com/matterport/Mask_RCNN) and [mtobeiyf's](https://github.com/mtobeiyf/keras-flask-deploy-webapp) repo


    
