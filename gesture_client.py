import cv2
import websocket
import json
import time # Included but not used in the main loop, kept for completeness if future delays are needed

# --- Configuration ---
# The WebSocket URL for the server stream
WEBSOCKET_URL = "ws://10.72.6.19:8000/ws/stream" 

class VideoStreamClient:
    """
    Manages the camera feed, WebSocket connection, and local video display 
    for real-time gesture/status feedback from a server.
    """
    
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.ws = None
        self.sos_alert_active = False

        # OpenCV Text Overlay Settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (0, 255, 0) # Green (BGR)
        self.thickness = 2
        self.line_type = cv2.LINE_AA

        # Variables for display text initialized to 'Connecting' state
        self.type_text = "Type: Connecting..."
        self.status_text = "Status: Connecting..."
        self.confidence_text = "Confidence: N/A"

    def __enter__(self):
        """Setup: Open camera and establish connection."""
        self._initialize_camera()
        self._connect_websocket()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup: Release resources regardless of success or failure."""
        self._cleanup()
        
    def _initialize_camera(self):
        """Opens the default camera (index 0)."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("🔴 Error: Could not open camera.")
            raise IOError("Camera failed to open.")

    def _connect_websocket(self):
        """Creates and establishes the WebSocket connection."""
        try:
            self.ws = websocket.create_connection(self.url)
            print(f"🟢 Connected to {self.url}")
        except ConnectionRefusedError:
            print("🔴 Error: Connection refused. Check if the server is running.")
            raise
        except Exception as e:
            print(f"🔴 An error occurred during connection: {e}")
            raise

    def _cleanup(self):
        """Releases the camera, closes the WebSocket, and destroys all OpenCV windows."""
        if self.ws and self.ws.connected:
            self.ws.close()
            print("⚫ WebSocket closed.")
            
        if self.cap:
            self.cap.release()
            print("⚫ Camera released.")

        cv2.destroyAllWindows()
        print("⚫ OpenCV windows closed.")

    def _process_server_feedback(self):
        """Receives and parses JSON feedback from the server, updating display variables."""
        try:
            # Attempt to receive a message from the server
            msg = self.ws.recv()
            feedback = json.loads(msg)
            # print(f"Feedback: {feedback}") # Uncomment for verbose logging

            # Check for SOS alert trigger
            if feedback.get('type') == 'sos_alert':
                self.sos_alert_active = True
                print("!!! RECEIVED SOS ALERT FROM SERVER !!!")

            # Update display variables safely
            msg_type = feedback.get('type', 'N/A')
            gesture = feedback.get('gesture', 'N/A')
            status = feedback.get('status', 'N/A')
            confidence = feedback.get('confidence', 0.0)

            # Format the text strings for display
            self.type_text = f"Type: {msg_type} ({gesture})"
            self.status_text = f"Status: {status}"
            self.confidence_text = f"Confidence: {confidence:.2f}%" 

        except websocket.WebSocketTimeoutException:
            # Ignore timeout and keep current display text
            pass
        except Exception as e:
            print(f"Error receiving feedback: {e}")
            self.status_text = "Status: Recv Error"
            # Re-raise to break the main loop
            raise

    def _draw_overlay_text(self, frame):
        """Draws the current status text and the '911' alert (if active) on the frame."""
        # Draw standard status texts
        cv2.putText(frame, self.type_text, (10, 30), self.font, self.font_scale, self.font_color, self.thickness, self.line_type)
        cv2.putText(frame, self.status_text, (10, 60), self.font, self.font_scale, self.font_color, self.thickness, self.line_type)
        cv2.putText(frame, self.confidence_text, (10, 90), self.font, self.font_scale, self.font_color, self.thickness, self.line_type)

        # Draw the BIG RED "911" alert if active
        if self.sos_alert_active:
            sos_font_scale = 5
            sos_font_color = (0, 0, 255) # Red (BGR)
            sos_thickness = 10
            sos_text = "911"
            
            # Calculate text size to center it
            text_size, _ = cv2.getTextSize(sos_text, self.font, sos_font_scale, sos_thickness)
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2

            cv2.putText(frame, sos_text, (text_x, text_y), self.font, sos_font_scale, sos_font_color, sos_thickness, self.line_type)

    def stream(self):
        """The main loop for capturing, sending, receiving, and displaying frames."""
        print("Starting video stream...")
        
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("⚠️ Cannot read frame from camera. Exiting.")
                break

            # 1. Encode the frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buf = cv2.imencode(".jpg", frame, encode_param)
            
            # 2. Send the frame bytes over WebSocket
            try:
                self.ws.send_binary(buf.tobytes())
            except Exception as e:
                print(f"Error sending frame: {e}")
                break

            # 3. Receive and process server feedback
            try:
                self._process_server_feedback()
            except:
                break # Exit loop on receive error

            # 4. Draw the status and alert overlay onto the frame
            self._draw_overlay_text(frame)

            # 5. Display the frame locally
            cv2.imshow('Client Camera Stream', frame)
            
            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# --- Main Execution ---
if __name__ == '__main__':
    # Use a context manager (with statement) to ensure proper cleanup
    try:
        with VideoStreamClient(WEBSOCKET_URL) as client:
            client.stream()
    except IOError:
        print("🛑 Client terminated due to camera failure.")
    except ConnectionRefusedError:
        print("🛑 Client terminated due to connection failure.")
    except Exception as e:
        print(f"🛑 Client terminated due to an unexpected error: {e}")