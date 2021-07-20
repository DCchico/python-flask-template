import io
from shapely.geometry import Point, Polygon
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
import numpy as np

class CoI:
    """
        Maintain Context of Intereset for each camera's field of view
    """
    def __init__(self, camera_name: str, cameras_config_path: str):
        self.isSet = False
        try:
            with open(cameras_config_path) as f:
                cameraconfig = json.load(f)
            self.coords = cameraconfig[camera_name]['coi']
            self.poly = Polygon(self.coords)
            self.isSet = True
        except:
            print("CoI is not set for Camera %s" % camera_name)

    def within(self, x: float, y: float) -> bool:
        """
        Check whether a point is within the COI
        """
        if not self.isSet:
            return True
        p = Point(x, y)
        return p.within(self.poly)


def load_frame_cpu(bytearr):
    image = Image.open(io.BytesIO(bytearr))
    # Explicitly load the image
    image.load()
    return image


def resize_frame(image, interpreter):
    width, height = common.input_size(interpreter)
    w, h = image.size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    return image.resize((w, h), Image.ANTIALIAS), (scale, scale)


def load_frame_tpu(image, interpreter):
    tensor = common.input_tensor(interpreter)
    tensor.fill(0)  # padding
    _, _, channel = tensor.shape
    w, h = image.size
    tensor[:h, :w] = np.reshape(image, (h, w, channel))


def inference(interpreter):
    interpreter.invoke()


def post_inference(interpreter, labels, scale, threshold, top_k, coi):
    objs = detect.get_objects(interpreter, threshold, scale)

    bbox_result = []
    for obj in objs:
        label = labels.get(obj.id, obj.id)
        if label in ['car', 'bus', 'truck']:
            mx = (obj.bbox.xmin + obj.bbox.xmax) / 2
            my = (obj.bbox.ymin + obj.bbox.ymax) / 2

            if coi.within(mx, my):
                bbox_result.append([obj.bbox.xmin, obj.bbox.ymin,
                                    obj.bbox.xmax, obj.bbox.ymax, obj.score])
    bbox_result.sort(key=lambda x: -x[4])
    return bbox_result[:top_k]


def handle(frame):
    """handle a request to the function
    Args:
        req (str): request body
    """
    labels = read_label_file("coco_labels.txt")
    interpreter = make_interpreter("ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
    interpreter.allocate_tensors()
    coi = CoI(None, None)

    image = load_frame_cpu(frame)
    resized_image, scale = resize_frame(image, interpreter)
    load_frame_tpu(resized_image, interpreter)
    inference(interpreter)
    bboxes = post_inference(interpreter, labels, scale, 0.2, 10, coi)
    return bboxes



if __name__ == '__main__':
    import cv2
    frame = cv2.imread("000000.jpeg")
    _, frame_enc = cv2.imencode('.jpeg', frame)
    img_string = frame_enc.tobytes()
    bb = handle(img_string)
    print("finished: " + str(bb))
