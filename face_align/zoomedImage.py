import cv2

class ZoomedImage:
    def __init__(self, target_size, margin_percent=5, smoothing_factor=0.1, threshold=20, frame_shape=None):
        self.target_size = target_size
        self.margin_percent = margin_percent
        self.smoothing_factor = smoothing_factor
        self.current_crop = None
        self.previous_crop = None
        self.threshold = threshold
        self.frame_shape = frame_shape
        
    def get_target_aspect(self):
        return self.target_size[0] / self.target_size[1]
    
    def calculate_initial_crop_dimensions(self, x1, y1, x2, y2):
        # Calculate box center and dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Add margin
        margin_x = int(box_width * self.margin_percent / 100)
        margin_y = int(box_height * self.margin_percent / 100)
        
        # Calculate initial crop dimensions
        crop_width = box_width + 2 * margin_x
        crop_height = box_height + 2 * margin_y
        
        return crop_width, crop_height
    
    
    def calculate_crop_dimensions(self, box):
        x1, y1, x2, y2 = map(int, box)
        
        crop_width, crop_height = self.calculate_initial_crop_dimensions(x1, y1, x2, y2)
        
        # Adjust dimensions to match target aspect ratio
        target_aspect = self.get_target_aspect()
        current_aspect = crop_width / crop_height
        if current_aspect > target_aspect:
            crop_height = int(crop_width / target_aspect)
        else:
            crop_width = int(crop_height * target_aspect)
            
        # Calculate crop coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        crop_x1 = max(0, center_x - crop_width // 2)
        crop_y1 = max(0, center_y - crop_height // 2)
        crop_x2 = min(self.frame_shape[1], center_x + crop_width // 2)
        crop_y2 = min(self.frame_shape[0], center_y + crop_height // 2)
        
        # Handle edge cases
        if crop_x1 == 0:
            crop_x2 = min(self.frame_shape[1], crop_x1 + crop_width)
        if crop_x2 == self.frame_shape[1]:
            crop_x1 = max(0, crop_x2 - crop_width)
        if crop_y1 == 0:
            crop_y2 = min(self.frame_shape[0], crop_y1 + crop_height)
        if crop_y2 == self.frame_shape[0]:
            crop_y1 = max(0, crop_y2 - crop_height)
            
        return (crop_x1, crop_y1, crop_x2, crop_y2)
    
    
    def smooth_crop_coordinates(self, new_crop):
        if self.current_crop is None:
            self.current_crop = new_crop
            return new_crop
        
        # Check if new crop is within threshold of current crop
        if all(abs(new_crop[i] - self.current_crop[i]) <= self.threshold 
               for i in range(4)):
            return self.current_crop
            
        # Smooth transition between crops
        smooth_crop = [
            int(self.current_crop[i] * (1 - self.smoothing_factor) + 
                new_crop[i] * self.smoothing_factor)
            for i in range(4)
        ]
        self.current_crop = smooth_crop
        return smooth_crop
    
    def process_frame(self, frame, box=None):
        if self.frame_shape is None:
            self.frame_shape = frame.shape
        
        if box is None:
            return cv2.resize(frame, self.target_size)
            
        # Calculate new crop dimensions
        new_crop = self.calculate_crop_dimensions(box)
        
        # Apply smoothing
        crop_x1, crop_y1, crop_x2, crop_y2 = self.smooth_crop_coordinates(new_crop)
        
        # Crop and resize
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        resized = cv2.resize(cropped, self.target_size)
        
        return resized
