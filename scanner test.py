########IMPORTANT#########
##IN ORDER FOR THIS TO WORK, YOU MUST INSTALL DEPENDANCIES AND CHOOSE AN APPROPRIATE SAVE FOLDER ON LINE 335
##########################
import cv2
import numpy as np
from pyzbar import pyzbar
import pytesseract
from PIL import Image
import os
import threading
import queue
import time
from datetime import datetime
import re

class PackageTracker:
    def __init__(self, camera_index=0, save_directory="captured_packages"):
        """
        Initialize the package tracking system
        
        Args:
            camera_index: Camera device index (usually 0 for default camera)
            save_directory: Directory to save captured images
        """
        self.camera_index = camera_index
        # Use os.path.normpath to handle Windows paths properly
        self.save_directory = os.path.normpath(save_directory)
        self.cap = None
        self.running = False
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
        print(f"Save directory created/verified: {self.save_directory}")
        
        # Queue for processing frames
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
        # Motion detection parameters
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        self.min_contour_area = 5000  # Minimum area for package detection
        
        # Tracking variables
        self.last_capture_time = 0
        self.capture_cooldown = 2.0  # Seconds between captures
        
    def initialize_camera(self):
        """Initialize camera with optimal settings for speed and quality"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
            
        print("Camera initialized successfully")
        
    def detect_package_motion(self, frame):
        """
        Detect if a package is present using motion detection
        
        Args:
            frame: Current video frame
            
        Returns:
            bool: True if package detected, False otherwise
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contour is large enough to be a package
        for contour in contours:
            if cv2.contourArea(contour) > self.min_contour_area:
                return True
                
        return False
        
    def preprocess_for_ocr(self, image):
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    def extract_tracking_number_pyzbar(self, image):
        """
        Extract tracking number using pyzbar (fastest for barcodes)
        
        Args:
            image: Input image
            
        Returns:
            str: Tracking number or None if not found
        """
        try:
            # Decode barcodes
            barcodes = pyzbar.decode(image)
            
            for barcode in barcodes:
                # Convert bytes to string
                barcode_data = barcode.data.decode('utf-8')
                
                # UPS tracking numbers are typically 18 characters starting with "1Z"
                if self.is_valid_ups_tracking(barcode_data):
                    return barcode_data
                    
        except Exception as e:
            print(f"Barcode detection error: {e}")
            
        return None
        
    def extract_tracking_number_ocr(self, image):
        """
        Extract tracking number using OCR as fallback
        
        Args:
            image: Input image
            
        Returns:
            str: Tracking number or None if not found
        """
        try:
            # Preprocess image
            processed = self.preprocess_for_ocr(image)
            
            # Use Tesseract with specific configuration for numbers
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Look for UPS tracking pattern
            tracking_pattern = r'1Z[A-Z0-9]{16}'
            matches = re.findall(tracking_pattern, text.replace(' ', ''))
            
            if matches:
                return matches[0]
                
            # Alternative pattern for other formats
            number_pattern = r'\b\d{12,18}\b'
            number_matches = re.findall(number_pattern, text.replace(' ', ''))
            
            if number_matches:
                return number_matches[-1]  # Return the last (likely bottom) number
                
        except Exception as e:
            print(f"OCR Error: {e}")
            
        return None
        
    def is_valid_ups_tracking(self, tracking_number):
        """
        Validate if the extracted text is a valid UPS tracking number
        
        Args:
            tracking_number: String to validate
            
        Returns:
            bool: True if valid UPS tracking number
        """
        # UPS tracking numbers start with "1Z" and are 18 characters total
        if len(tracking_number) == 18 and tracking_number.startswith('1Z'):
            return True
            
        # Alternative formats (10-digit, 12-digit, etc.)
        if len(tracking_number) >= 10 and tracking_number.isdigit():
            return True
            
        return False
        
    def capture_and_save_package(self, frame):
        """
        Capture package image and save with tracking number as filename
        
        Args:
            frame: Video frame containing the package
        """
        current_time = time.time()
        
        # Implement cooldown to prevent duplicate captures
        if current_time - self.last_capture_time < self.capture_cooldown:
            return
            
        # Try barcode detection first (fastest)
        tracking_number = self.extract_tracking_number_pyzbar(frame)
        
        # Fallback to OCR if barcode detection fails
        if not tracking_number:
            tracking_number = self.extract_tracking_number_ocr(frame)
            
        if tracking_number:
            # Clean tracking number for filename (remove invalid characters)
            clean_tracking = re.sub(r'[<>:"/\\|?*]', '_', tracking_number)
            
            # Generate filename with timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_tracking}_{timestamp}.jpg"
            
            # Use os.path.join for proper path construction
            filepath = os.path.join(self.save_directory, filename)
            
            # Save high-quality image
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                print(f"Package captured: {filename}")
                print(f"Saved to: {filepath}")
            else:
                print(f"Failed to save image to: {filepath}")
                
            self.last_capture_time = current_time
        else:
            print("Could not extract tracking number from package")
            
    def process_frame(self, frame):
        """
        Process a single frame for package detection and capture
        
        Args:
            frame: Video frame to process
        """
        # Detect if package is present
        if self.detect_package_motion(frame):
            # Add frame to processing queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
                
    def processing_worker(self):
        """
        Worker thread for processing captured frames
        """
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                self.capture_and_save_package(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                
    def start_tracking(self):
        """
        Start the package tracking system
        """
        try:
            self.initialize_camera()
            self.running = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("Package tracking started. Press 'q' to quit.")
            print(f"Images will be saved to: {self.save_directory}")
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame")
                    break
                    
                # Process frame for package detection
                self.process_frame(frame)
                
                # Display live feed (optional - remove for headless operation)
                cv2.imshow('Package Tracker', frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping package tracker...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.stop_tracking()
            
    def stop_tracking(self):
        """
        Stop the package tracking system
        """
        self.running = False
        
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        print("Package tracking stopped")

# Usage example with CORRECTED Windows path
if __name__ == "__main__":
    # Method 1: Use raw string (prefix with r)
    tracker = PackageTracker(
        camera_index=0,
        save_directory=r"C:\Users\user\Desktop\savefolder"
    )
    
    # Start tracking
    tracker.start_tracking()
