import aiohttp
from aiohttp import web
import ssl
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2
import torch

# Define the device (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with the path to your YOLOv8 model
model.to(device)  # Move the model to the specified device

# Wrap model in DataParallel if more than one GPU is available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

async def index(request):
    return web.FileResponse('index.html')

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['websockets'].append(ws)

    async def send_detection(img_bytes):
        try:
            # Load and preprocess the image
            img = Image.open(io.BytesIO(img_bytes))
            img_np = np.array(img)

            # Save original size
            orig_size = img.size
            orig_width, orig_height = orig_size

            # Resize image to 640x640 for model input
            img_resized = cv2.resize(img_np, (640, 640))
            img_resized = img_resized.transpose((2, 0, 1))  # Change to (C, H, W)
            img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

            # Convert to tensor and move to device
            img_tensor = torch.from_numpy(img_resized).float().to(device) / 255.0

            # Perform prediction using the base model inside DataParallel
            if isinstance(model, torch.nn.DataParallel):
                results = model.module.predict(source=img_tensor, save=False, show=False)
                model_names = model.module.names
            else:
                results = model.predict(source=img_tensor, save=False, show=False)
                model_names = model.names

            # Visualize results (if needed)
            img_with_boxes = results[0].plot()

            # Convert image to RGB
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

            # Resize image to fit the original dimensions
            target_width, target_height = orig_width, orig_height
            img_with_boxes_resized = cv2.resize(img_with_boxes, (target_width, target_height))

            # Convert image to JPEG bytes
            _, img_encoded = cv2.imencode('.jpg', img_with_boxes_resized)
            img_bytes = img_encoded.tobytes()

            # Create summary text
            summary_text = "Detected objects:\n"
            for item in results[0].boxes:
                class_id = int(item.cls)
                bbox = item.xyxy[0]
                summary_text += f"{model_names[class_id]}: {bbox.tolist()}\n"

            # Combine image and text
            response_data = {
                "image": base64.b64encode(img_bytes).decode('utf-8'),
                "summary": summary_text
            }

            # Send processed image and summary text back to clients
            for client in request.app['websockets']:
                await client.send_json(response_data)

        except Exception as e:
            print(f"Error processing image: {e}")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                await send_detection(msg.data)

    except Exception as e:
        print(f"WebSocket connection closed with exception: {e}")

    finally:
        request.app['websockets'].remove(ws)
        print('WebSocket connection closed')

app = web.Application()
app['websockets'] = []
app.add_routes([web.get('/', index), web.get('/ws', websocket_handler)])

if __name__ == '__main__':
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('/home/gushole1/pyprojects/webrtcAI/mkcerts/server.crt', '/home/gushole1/pyprojects/webrtcAI/mkcerts/server.key')
    web.run_app(app, ssl_context=ssl_context)
