"""
SCRFD Face Detector - Laptop/CPU Version
Adapted from scrfd_detector_board.py (NPU version)
"""

import numpy as np
import cv2
import tensorflow as tf


class SCRFDDetector:
    def __init__(self, model_path="scrfd_500m_full_int8.tflite"):
        """Initialize SCRFD detector for CPU"""
        print(f"[SCRFD-CPU] Loading model: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        
        # Input shape
        self.input_height = self.input_details['shape'][1]
        self.input_width = self.input_details['shape'][2]
        
        # Quantization params
        self.input_scale, self.input_zero = self.input_details['quantization']
        
        print(f"[SCRFD-CPU] Model loaded: {self.input_height}x{self.input_width}")
        print(f"[SCRFD-CPU] Input: scale={self.input_scale}, zero={self.input_zero}")
    
    def preprocess(self, image):
        """Preprocess image for SCRFD"""
        # Resize to model input size
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Quantize for INT8 model
        if self.input_scale > 0:
            img_quantized = (img_float / self.input_scale) + self.input_zero
            img_quantized = np.clip(img_quantized, -128, 127).astype(np.int8)
        else:
            img_quantized = img_float
        
        return np.expand_dims(img_quantized, axis=0)
    
    def detect(self, image, score_threshold=0.45):
        """Detect faces in image"""
        h, w = image.shape[:2]
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()
        
        # Get all outputs and dequantize
        outputs = []
        for out_detail in self.output_details:
            output_data = self.interpreter.get_tensor(out_detail['index'])
            scale, zero_point = out_detail['quantization']
            
            if scale > 0:
                output_float = (output_data.astype(np.float32) - zero_point) * scale
            else:
                output_float = output_data.astype(np.float32)
            
            outputs.append(output_float)
        
        # Decode SCRFD outputs with anchor-based decoding
        detections = self._decode_scrfd_output(outputs, score_threshold)
        
        # Scale back to original frame size and convert to landmarks format
        for det in detections:
            det['box'][0] *= w / self.input_width   # x1
            det['box'][1] *= h / self.input_height  # y1
            det['box'][2] *= w / self.input_width   # x2
            det['box'][3] *= h / self.input_height  # y2
            
            # Scale keypoints
            det['keypoints'][0::2] *= w / self.input_width   # x coords
            det['keypoints'][1::2] *= h / self.input_height  # y coords
        
        return detections
    
    def _decode_scrfd_output(self, outputs, score_threshold):
        """Decode SCRFD multi-scale anchor-based outputs"""
        # Output mapping
        boxes_stride16 = outputs[0]     # [600, 4]
        kps_stride16 = outputs[5]       # [600, 10]
        scores_stride16 = outputs[6]    # [600, 1]
        
        scores_stride32 = outputs[1]    # [160, 1]
        kps_stride32 = outputs[2]       # [160, 10]
        boxes_stride32 = outputs[7]     # [160, 4]
        
        boxes_stride8 = outputs[3]      # [2400, 4]
        scores_stride8 = outputs[4]     # [2400, 1]
        kps_stride8 = outputs[8]        # [2400, 10]
        
        # Generate anchors
        h32, w32 = int(np.ceil(self.input_height / 32)), int(np.ceil(self.input_width / 32))
        h16, w16 = int(np.ceil(self.input_height / 16)), int(np.ceil(self.input_width / 16))
        h8, w8 = int(np.ceil(self.input_height / 8)), int(np.ceil(self.input_width / 8))
        
        anchors_stride32 = self._generate_anchors((h32, w32), stride=32)
        anchors_stride32 = np.repeat(anchors_stride32, 2, axis=0)
        
        anchors_stride16 = self._generate_anchors((h16, w16), stride=16)
        anchors_stride16 = np.repeat(anchors_stride16, 2, axis=0)
        
        anchors_stride8 = self._generate_anchors((h8, w8), stride=8)
        anchors_stride8 = np.repeat(anchors_stride8, 2, axis=0)
        
        # Decode boxes from anchor + distance format
        boxes_stride8_decoded = self._distance2bbox(anchors_stride8, boxes_stride8, stride=8)
        boxes_stride16_decoded = self._distance2bbox(anchors_stride16, boxes_stride16, stride=16)
        boxes_stride32_decoded = self._distance2bbox(anchors_stride32, boxes_stride32, stride=32)
        
        # Decode keypoints from anchor + distance format
        kps_stride8_decoded = self._distance2kps(anchors_stride8, kps_stride8, stride=8)
        kps_stride16_decoded = self._distance2kps(anchors_stride16, kps_stride16, stride=16)
        kps_stride32_decoded = self._distance2kps(anchors_stride32, kps_stride32, stride=32)
        
        # Combine all scales
        all_boxes = np.vstack([boxes_stride32_decoded, boxes_stride16_decoded, boxes_stride8_decoded])
        all_scores = np.vstack([scores_stride32, scores_stride16, scores_stride8])
        all_kps = np.vstack([kps_stride32_decoded, kps_stride16_decoded, kps_stride8_decoded])
        
        # Filter by score
        valid_mask = all_scores[:, 0] > score_threshold
        filtered_boxes = all_boxes[valid_mask]
        filtered_scores = all_scores[valid_mask, 0]
        filtered_kps = all_kps[valid_mask]
        
        if len(filtered_boxes) == 0:
            return []
        
        # NMS
        keep_indices = self._nms(filtered_boxes, filtered_scores, iou_threshold=0.4)
        
        detections = []
        for idx in keep_indices:
            box = filtered_boxes[idx]
            kps = filtered_kps[idx]
            
            # Validate detection
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            
            if box_w < 15 or box_h < 15:
                continue
            if box_w > self.input_width * 0.95 or box_h > self.input_height * 0.95:
                continue
            
            aspect_ratio = box_h / (box_w + 1e-6)
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                continue
            
            detections.append({
                'box': box.copy(),
                'score': filtered_scores[idx],
                'keypoints': kps.copy()
            })
        
        # Sort by score and keep best
        detections.sort(key=lambda x: x['score'], reverse=True)
        return detections[:1]  # Return only best detection
    
    def _generate_anchors(self, fmap_size, stride):
        """Generate anchor centers"""
        anchors = []
        for y in range(fmap_size[0]):
            for x in range(fmap_size[1]):
                anchors.append([x * stride, y * stride])
        return np.array(anchors, dtype=np.float32)
    
    def _distance2bbox(self, points, distance, stride):
        """Decode boxes from anchor + distance format"""
        x1 = points[:, 0] - distance[:, 0] * stride
        y1 = points[:, 1] - distance[:, 1] * stride
        x2 = points[:, 0] + distance[:, 2] * stride
        y2 = points[:, 1] + distance[:, 3] * stride
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _distance2kps(self, points, distance, stride):
        """Decode keypoints from anchor + distance format"""
        kps = []
        for i in range(5):
            px = points[:, 0] + distance[:, i*2] * stride
            py = points[:, 1] + distance[:, i*2+1] * stride
            kps.extend([px, py])
        return np.stack(kps, axis=-1)
    
    def _nms(self, boxes, scores, iou_threshold=0.3):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            union = areas[i] + areas[order[1:]] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union!=0)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


def align_face(image, landmarks, target_size=112):
    """Align face using 5-point landmarks"""
    # Standard reference points for 112x112 face
    scale = target_size / 112.0
    reference = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose
        [41.5493, 92.3655],  # Left mouth
        [70.7299, 92.2041]   # Right mouth
    ], dtype=np.float32) * scale
    
    # landmarks: [5, 2] array or flat [10] array
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(5, 2)
    
    landmarks = landmarks.astype(np.float32)
    
    # Estimate similarity transform
    tform = cv2.estimateAffinePartial2D(landmarks, reference)[0]
    
    # Apply transform
    aligned = cv2.warpAffine(image, tform, (target_size, target_size), 
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return aligned, tform
