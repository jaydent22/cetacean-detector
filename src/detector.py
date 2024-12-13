import os
from tqdm import tqdm
from base_model import BaseModel
from truncated_result import TruncatedResult

class Detector(object):
    """An object detector for detecting cetaceans."""

    def __init__(self, weights=None, version="yolov8", device="cpu"):
        """Initializes the Detector model.

        Args:
            weights (str, optional): 
                Path to the weights file. Defaults to None.
            version (str, optional):
                Version of pretrained YOLO model to load. Ignored if weights is passed. Defaults to "yolov8".
            device (str, optional):
                Device for inference. Defaults to cpu.
        """

        try:
            if weights is not None:
                self.model = BaseModel(weights)
            else:
                self.model = BaseModel(f"{version}.pt")
        except:
            self.model = BaseModel(f"{version}.pt")
        self.device = device

    
    def detect_single(self, path, conf=0.25):
        """Runs object detection on a single image.

        Args:
            path (str): 
                Path to the image.
            conf (int, optional):
                Minimum confidence threshold for detections. Defaults to 0.25.

        Returns:
            ultralytics.engine.results.Results: An Ultralytics Results object.
        """

        # predict() returns an array of predictions
        result = self.model.predict(path, device=self.device, conf=conf)[0]

        result = TruncatedResult(result)

        return result

    def detect_batch(self, imgs=None, path=None, conf=0.25, batch_size=1):
        """Runs object detection on a directory of images.

        Args:
            imgs (list, optional):
                List of paths to images. Both path and imgs cannot be passed. Defaults to None.
            path (str, optional): 
                Path to a directory of images. Ignored if imgs is passed. Both path and imgs cannot be passed. Defaults to None.
            conf (int, optional):
                Minimum confidence threshold for detections. Defaults to 0.25.
            batch_size (int, optional):
                Batch size for inference. A larger batch size operates on more images at once, decreasing inference time.

        Returns:
            list: A list containing dictionaries for each image. A dictionary contains the path, image name, boxes in specified format, and confidence scores per box.

        Raises:
            ValueError: Raises an exception if both imgs and path are passed.
        """
        if imgs is not None and path is not None:
            raise ValueError("Only one of either imgs or path can be passed")

        results_list = []
        self.imgs = []
        
        if imgs is not None:
            self.imgs = imgs
        elif path is not None:
            num_imgs = sum(len(files) for _, _, files in os.walk(path))
            with tqdm(total=num_imgs, desc="Extracting images") as pbar:
                for root, _, files in os.walk(path):
                    for file in files:
                        pbar.update(1)
                        self.imgs.append(os.path.join(root, file))

        results = self.model.predict(self.imgs, device=self.device, conf=conf, batch=batch_size)

        for result in results:
            trunc = TruncatedResult(result)
            results_list.append(trunc)

        return results_list