import os
class TruncatedResult:
    """A TruncatedResult object to extract information of interest from an Ultralytics Results object.

    TruncatedResult includes image path, image id, and bounding boxes.
    """
    def __init__(self, result):
        """Initializes a TruncatedResult.

        boxes attribute contains coordinates in both xyxy and xywh, and corresponding confidence scores.

        Args:
            result (ultralytics.engine.results.Results): Ultralytics YOLO object detection result object.
        """
        self.path = result.path
        self.img_id = os.path.basename(self.path)
        self.boxes = result.boxes.cpu().numpy()

    def __format_boxes(self, boxes):
        formatted_boxes = ""
        for i, box in enumerate(boxes):
            if i > 0:
                formatted_boxes += " " * 11
            formatted_boxes += f"{box}"
            if i < len(boxes) - 1:
                formatted_boxes += "\n"
        return formatted_boxes
        

    
    def __str__(self):
        return f"""
path: {self.path}
img_id: {self.img_id}
boxes:
    conf: {self.boxes.conf}
    xyxy: [{self.__format_boxes(self.boxes.xyxy)}]
    xywh: [{self.__format_boxes(self.boxes.xywh)}]
"""