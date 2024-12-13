from ultralytics import YOLO


class BaseModel(YOLO):

    def __init__(self, weights=None):
        """Initializes the Detector model.

        Args:
            weights (str): 
                Path to the weights file.
        """
        super(BaseModel, self).__init__(weights)