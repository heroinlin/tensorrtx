"""

This module implements the TrtYolov5 class.
"""


import ctypes

import numpy as np
import cv2
import random
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import time


def py_cpu_nms(boxes, scores,  thresh):
    """Pure Python NMS baseline."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class TrtInference(object):
    def __init__(self, model_path=None, cuda_ctx=None):
        self.model_path = model_path
        if model_path is None:
            print("please set trt model path!")
            exit()
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx is None:
            self.cuda_ctx = cuda.Device(0).make_context()
        if self.cuda_ctx:
            self.cuda_ctx.push()
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def _load_plugins(self):
        pass

    def _load_engine(self):
        with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def destroy(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream
        if self.cuda_ctx:
            self.cuda_ctx.pop()
            del self.cuda_ctx

    def inference(self, img):
        np.copyto(self.host_inputs[0], img.ravel())
        if self.cuda_ctx:
            self.cuda_ctx.push()
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        output = self.host_outputs[0]
        return output


class TrtYolov5(TrtInference):
    """TrtYolov5 class encapsulates things needed to run TRT Yolov5."""
    def __init__(self, model_path=None, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model_path = model_path
        self.input_shape = (608, 608)
        super(TrtYolov5, self).__init__(model_path, cuda_ctx)
        self.configs = {
            "conf_th": 0.3,
            "iou_th": 0.3
        }

    def _load_plugins(self):
        PLUGIN_LIBRARY = "build/libmyplugins.so"
        ctypes.CDLL(PLUGIN_LIBRARY)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        height, width = self.input_shape
        r_w = width / origin_w
        r_h = height / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (height - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (height - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (width - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (width - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y
    
    def _preprocess_trt(self, img, input_shape=(608, 608)):
        """Preprocess an image before TRT Yolov5 inferencing."""
        h, w, c = img.shape
        height, width = input_shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Calculate width and height and paddings
        r_w = width / w
        r_h = height / h
        if r_h > r_w:
            tw = width
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((height - th) / 2)
            ty2 = height - th - ty1
        else:
            tw = int(r_h * w)
            th = height
            tx1 = int((width - tw) / 2)
            tx2 = width - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image

    def _postprocess_trt(self, img, output, conf_th, iou_th):
        """Postprocess TRT Yolov5 output."""
        origin_h, origin_w = img.shape[:2]
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]

        # Choose those boxes that score > CONF_THRESH
        si = scores > conf_th
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]

        boxes = self._xywh2xyxy(origin_h, origin_w, boxes)
        indices = py_cpu_nms(boxes, scores, iou_th)
        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        if result_boxes is not None:
            result_boxes[:, [0, 2]] /= origin_w
            result_boxes[:, [1, 3]] /= origin_h
        return result_boxes, result_scores, result_classid

    def detect(self, img):
        """Detect objects in the input image."""
        img_resized = self._preprocess_trt(img, self.input_shape)
        pred = self.inference(img_resized)
        output = self._postprocess_trt(img, pred, self.configs["conf_th"], self.configs["iou_th"])
        # draw_detection_rects(img, np.hstack([output[0], 
        #                                      np.expand_dims(output[1], axis=1), 
        #                                      np.expand_dims(output[2], axis=1)]))
        # cv2.imshow("detect", img)
        # cv2.waitKey(0)
        return output


class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        output = self.func(*self.args)
        # print(output)
        # print(output.shape)
        return output


def draw_detection_rects(image: np.ndarray, detection_rects: np.ndarray, colors=None, names=None, method=1):
    if not isinstance(detection_rects, np.ndarray):
        detection_rects = np.array(detection_rects)
    if method:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = 1.0
        height = 1.0
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]
    for index in range(detection_rects.shape[0]):
        if detection_rects.shape[1] == 6:
            if len(colors) <= int(detection_rects[index, 5]):
                colors.extend([[random.randint(0, 255) for _ in range(3)] for _ in range(int(detection_rects[index, 5]) - len(colors) + 2)])
            color = colors[int(detection_rects[index, 5])]
            thickness = 2
        else:
            color = colors[0]
            thickness = 2
        cv2.rectangle(image,
                      (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                      (int(detection_rects[index, 2] * width), int(detection_rects[index, 3] * height)),
                      color,
                      thickness=thickness)
        if detection_rects.shape[1] >= 5:
            if detection_rects.shape[1] == 6:
                label_id = int(detection_rects[index, 5])
                if names is not None and label_id < len(names):
                    msg = f"{names[label_id]}: {detection_rects[index, 4]:.03f}"
                else:
                    msg = f"{label_id}: {detection_rects[index, 4]:.03f}"
            else:
                msg = f"{detection_rects[index, 4]:.03f}"
            cv2.putText(image, msg,
                        (int(detection_rects[index, 0] * width), int(detection_rects[index, 1] * height)),
                        1, 1, color)
    return image


def compute_time(func, args, run_num=100):
    start_time = time.time()
    for i in range(run_num):
        func(*args)
    end_time = time.time()
    avg_run_time = (end_time - start_time)*1000/run_num
    return avg_run_time


def compute_inference_time():
    engine_file_path = "weights/yolov5l.engine"
    detector = TrtYolov5(model_path=engine_file_path)

    input_image_paths = ["data/images/bus.jpg"]
    input_image_path = input_image_paths[0]

    image = cv2.imread(input_image_path)
    input_image = detector._preprocess_trt(image)
    pred = detector.inference(input_image)
    outputs = detector._postprocess_trt(image, pred, 0.3, 0.3)
    print(outputs)

    preprocess_time = compute_time(detector._preprocess_trt, [image])
    print("avg preprocess time is {:02f} ms".format(preprocess_time))

    inference_time = compute_time(detector.inference, [input_image])
    print("avg inference time is {:02f} ms".format(inference_time))

    postprocess_time = compute_time(detector._postprocess_trt, [image, pred, 0.3, 0.3])
    print("avg postprocess time is {:02f} ms".format(postprocess_time))

    total_time = compute_time(detector.detect, [image])
    print("avg total predict time is {:02f} ms".format(total_time))

    detector.destroy()


def main():
    engine_file_path = "weights/yolov5l_int8.engine"
    yolov5_wrapper = TrtYolov5(model_path=engine_file_path)

    input_image_paths = ["data/images/bus.jpg"]
    
    inference_num = 100
    input_image_path = input_image_paths[0]
    image = cv2.imread(input_image_path)
    thread1 = myThread(yolov5_wrapper.detect, [image])
    thread1.start()
    thread1.join()
    start_time = time.time()
    for i in range(inference_num):
        # create a new thread to do inference
        image = cv2.imread(input_image_path)
        thread1 = myThread(yolov5_wrapper.detect, [image])
        thread1.start()
        thread1.join()
    end_time = time.time()
    print("avg predict time is {:02f} ms".format((end_time-start_time)*1000/inference_num))
    # destroy the instance
    yolov5_wrapper.destroy()


if __name__ == "__main__":
    # main()
    compute_inference_time()