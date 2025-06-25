from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import time
import shutil
import glob
import datetime
import torch
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np
import cv2
import face_recognition
from PIL import Image as pImage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re
import logging
import base64

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/heatmaps", exist_ok=True)
os.makedirs("static/uploaded_images", exist_ok=True)
os.makedirs("uploaded_videos", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def detect_and_crop_face(frame, padding=40, min_face_size=100):
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None, None
    
    face_sizes = [(bottom - top, right - left) for top, right, bottom, left in face_locations]
    largest_idx = np.argmax([w * h for h, w in face_sizes])
    top, right, bottom, left = face_locations[largest_idx]
    
    face_height = bottom - top
    face_width = right - left
    if face_height < min_face_size or face_width < min_face_size:
        return None, None
    
    top_pad = max(0, top - padding)
    bottom_pad = min(frame.shape[0], bottom + padding)
    left_pad = max(0, left - padding)
    right_pad = min(frame.shape[1], right + padding)
    
    return frame[top_pad:bottom_pad, left_pad:right_pad, :], (top, right, bottom, left)

def draw_face_rectangles(frame, face_coords, is_real=True):
    top, right, bottom, left = face_coords
    color = (0, 255, 0) if is_real else (0, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = "Face"
    cv2.putText(frame, label, (left, top - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, video_names, sequence_length=40, transform=None, min_face_size=100):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
        self.min_face_size = min_face_size

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        valid_frames = []
        padding = 40
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            cropped_frame, _ = detect_and_crop_face(frame, padding, self.min_face_size)
            if cropped_frame is not None:
                valid_frames.append(cropped_frame)
                if len(valid_frames) == self.count:
                    break
        
        if len(valid_frames) < self.count and valid_frames:
            last_valid = valid_frames[-1]
            valid_frames += [last_valid] * (self.count - len(valid_frames))
        
        for frame in valid_frames:
            try:
                frames.append(self.transform(frame))
            except:
                frame_pil = transforms.ToPILImage()(frame)
                frame_tensor = transforms.Resize((im_size, im_size))(frame_pil)
                frame_tensor = transforms.ToTensor()(frame_tensor)
                frames.append(frame_tensor)
                
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def allowed_video_file(filename):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_VIDEO_EXTENSIONS

def get_accurate_model(sequence_length):
    list_models = glob.glob(os.path.join("models", "*.pt"))
    target_seq_str = str(sequence_length)
    candidate_models = []
    
    for model_path in list_models:
        filename = os.path.basename(model_path)
        numbers = re.findall(r'\d+', filename)
        if target_seq_str in numbers:
            candidate_models.append(filename)
    
    if candidate_models:
        return candidate_models[0]
    else:
        for model_path in list_models:
            filename = os.path.basename(model_path)
            if "40" in filename:
                return filename
    return None

def im_convert(tensor, video_file_name=""):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def generate_gradcam_heatmap(model, img, video_file_name=""):
    fmap, logits = model(img)
    logits_softmax = sm(logits)
    confidence, prediction = torch.max(logits_softmax, 1)
    confidence_val = confidence.item() * 100
    pred_idx = prediction.item()
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    fmap_last = fmap[-1].detach().cpu().numpy()
    nc, h, w = fmap_last.shape
    fmap_reshaped = fmap_last.reshape(nc, h*w)
    heatmap_raw = np.dot(fmap_reshaped.T, weight_softmax[pred_idx, :].T)
    heatmap_raw -= heatmap_raw.min()
    heatmap_raw /= heatmap_raw.max()
    heatmap_img = np.uint8(255 * heatmap_raw.reshape(h, w))
    heatmap_resized = cv2.resize(heatmap_img, (im_size, im_size))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    original_img = im_convert(img[:, -1, :, :, :])
    original_img_uint8 = (original_img * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_img_uint8, 0.6, heatmap_colored, 0.4, 0)
    heatmap_filename = f"{video_file_name}_heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
    cv2.imwrite(heatmap_path, overlay)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Frame')
    plt.axis('on')
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title('Attention Heatmap')
    plt.axis('on')
    plt.subplot(1, 3, 3)
    plt.imshow(overlay[..., ::-1])
    plt.title(f'Overlay - Prediction: {"REAL" if pred_idx == 1 else "FAKE"} ({confidence_val:.1f}%)')
    plt.axis('on')
    plt.tight_layout()
    plt_filename = f"{video_file_name}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt_path = os.path.join("static", "heatmaps", plt_filename)
    plt.savefig(plt_path, dpi=150, bbox_inches='tight')
    plt.close()
    return {
        'prediction': pred_idx,
        'confidence': confidence_val,
        'heatmap_path': f"/static/heatmaps/{heatmap_filename}",
        'analysis_path': f"/static/heatmaps/{plt_filename}"
    }

def predict_with_gradcam(model, img, video_file_name=""):
    return generate_gradcam_heatmap(model, img, video_file_name)

@app.post("/api/upload")
async def api_upload_video(file: UploadFile = File(...)):
    if not allowed_video_file(file.filename):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    file_ext = file.filename.split('.')[-1]
    saved_video_file = f'uploaded_video_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.{file_ext}'
    file_path = os.path.join("uploaded_videos", saved_video_file)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Video saved to: {file_path}")
    result = await process_video(file_path)
    return {
        "status": "success",
        "result": result["output"],
        "confidence": result["confidence"],
        "frames_processed": result["frames_used"],
        "annotated_images": result["annotated_images"],
        "faces_cropped_images": result["faces_cropped_images"],
        "heatmap_image": result["heatmap_image"],
        "analysis_image": result["analysis_image"],
        "gradcam_explanation": result["gradcam_explanation"]
    }

async def process_video(video_file):
    try:
        if not os.path.exists(video_file):
            raise HTTPException(status_code=400, detail="Video file not found")

        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        target_faces = 40
        min_faces_required = 30
        max_frames_to_scan = 80
        static_uploaded_images_dir = os.path.join("static", "uploaded_images")
        os.makedirs(static_uploaded_images_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        valid_faces = []
        face_coords_list = []
        original_frames = []
        faces_cropped_images = []
        
        while cap.isOpened() and frame_count < max_frames_to_scan:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            cropped_frame, face_coords = detect_and_crop_face(frame, padding=0)
            if cropped_frame is None:
                continue
            valid_faces.append(cropped_frame)
            face_coords_list.append(face_coords)
            original_frames.append(frame.copy())
            cropped_face_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            cropped_face_img = pImage.fromarray(cropped_face_rgb, 'RGB')
            cropped_image_name = f"{video_file_name_only}_cropped_face_{len(valid_faces)}.png"
            cropped_image_path = os.path.join(static_uploaded_images_dir, cropped_image_name)
            cropped_face_img.save(cropped_image_path)
            faces_cropped_images.append(f"/static/uploaded_images/{cropped_image_name}")
            
            print(f"Found face in frame {frame_count} - total faces: {len(valid_faces)}")
            
            if len(valid_faces) == target_faces:
                break
        cap.release()
        
        if len(valid_faces) < min_faces_required:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient frames with detectable faces found in {frame_count} frames (minimum {min_faces_required} required)"
            )
        
        if len(valid_faces) >= target_faces:
            processed_frames = valid_faces[:target_faces]
            frames_used = target_faces
        else:
            processed_frames = valid_faces
            frames_used = len(valid_faces)
        
        logger.info(f"Used {frames_used} faces for prediction")
        transformed_frames = [train_transforms(frame) for frame in processed_frames]
        frames_tensor = torch.stack(transformed_frames)
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        model = Model(2).to(device)
        model_filename = get_accurate_model(frames_used) or get_accurate_model(40)
        if not model_filename:
            raise HTTPException(
                status_code=500, 
                detail=f"No suitable model found for sequence length {frames_used} or 40"
            )
        model_path = os.path.join("models", model_filename)
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500, 
                detail=f"Model file not found at {model_path}"
            )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        gradcam_result = predict_with_gradcam(model, frames_tensor, video_file_name_only)
        confidence = round(gradcam_result['confidence'], 1)
        output = "REAL" if gradcam_result['prediction'] == 1 else "FAKE"
        is_real = output == "REAL"
        annotated_images = []
        for idx, (frame, face_coords) in enumerate(zip(original_frames, face_coords_list)):
            annotated_frame = draw_face_rectangles(frame.copy(), face_coords, is_real)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            annotated_img = pImage.fromarray(annotated_frame_rgb, 'RGB')
            annotated_image_name = f"{video_file_name_only}_annotated_{idx+1}.png"
            annotated_image_path = os.path.join(static_uploaded_images_dir, annotated_image_name)
            annotated_img.save(annotated_image_path)
            annotated_images.append(f"/static/uploaded_images/{annotated_image_name}")
        gradcam_explanation = {
            "description": "The heatmap shows areas where the AI model focused its attention when making the prediction.",
            "interpretation": {
                "red_areas": "High attention - areas that strongly influenced the decision",
                "yellow_areas": "Medium attention - moderately important areas", 
                "blue_areas": "Low attention - areas with minimal influence on the decision"
            },
            "prediction_basis": f"The model classified this video as {output} with {confidence}% confidence based on the highlighted facial regions."
        }
        return {
            "annotated_images": annotated_images,
            "faces_cropped_images": faces_cropped_images,
            "output": output,
            "confidence": confidence,
            "frames_used": frames_used,
            "heatmap_image": gradcam_result['heatmap_path'],
            "analysis_image": gradcam_result['analysis_path'],
            "gradcam_explanation": gradcam_explanation
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing video: {str(e)}"
        )

@app.post("/predict")
async def predict_frames(data: dict):
    try:
        print("Received request to /predict endpoint")
        frames = data.get('frames', [])
        if not frames:
            print("No frames provided in request")
            raise HTTPException(status_code=400, detail="No frames provided")

        print(f"Processing {len(frames)} frames")
        target_frames = 40
        min_frames_required = 30  # Increased from 20 to 30
        max_frames_to_scan = 80
        valid_faces = []
        
        frames_to_process = frames[:max_frames_to_scan]
        print(f"Scanning {len(frames_to_process)} frames for faces")
        
        for i, frame_base64 in enumerate(frames_to_process):
            try:
                if ',' in frame_base64:
                    frame_base64 = frame_base64.split(',')[1]
                
                frame_data = base64.b64decode(frame_base64)
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if frame is None:
                    print(f"Frame {i+1} is None, skipping")
                    continue
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cropped_frame, _ = detect_and_crop_face(frame)
                if cropped_frame is None:
                    continue
                    
                valid_faces.append(cropped_frame)
                print(f"Found face in frame {i+1} - total faces: {len(valid_faces)}")
                
                if len(valid_faces) == target_frames:
                    break
                
            except Exception as e:
                print(f"Error processing frame {i+1}: {str(e)}")
                continue

        print(f"Found {len(valid_faces)} frames with detectable faces in {max_frames_to_scan} frames")
        
        if len(valid_faces) < min_frames_required:
            return {
                "error": "insufficient_faces",
                "message": f"Insufficient faces detected (minimum {min_frames_required} required, found {len(valid_faces)})"
            }

        processed_frames = valid_faces
        actual_sequence_length = len(processed_frames)
        print(f"Using {actual_sequence_length} faces for prediction")

        frames_tensor = torch.stack([
            train_transforms(frame) for frame in processed_frames
        ])
        frames_tensor = frames_tensor.unsqueeze(0).to(device)  # Use device

        model = Model(2).to(device)  # Use device
        model_filename = get_accurate_model(actual_sequence_length)
        
        if not model_filename:
            print(f"No suitable model found for sequence length {actual_sequence_length}, trying 40")
            model_filename = get_accurate_model(40)
        
        if not model_filename:
            print(f"No suitable model found for sequence lengths {actual_sequence_length} or 40")
            raise HTTPException(
                status_code=500,
                detail=f"No suitable model found for sequence lengths {actual_sequence_length} or 40"
            )
        
        print(f"Using model: {model_filename} for sequence length {actual_sequence_length}")
        
        model_path = os.path.join("models", model_filename)
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))  # Use device
        model.eval()

        with torch.no_grad():
            _, logits = model(frames_tensor)
            probabilities = sm(logits)
            _, prediction = torch.max(probabilities, 1)
            confidence = probabilities[:, int(prediction.item())].item() * 100
            
            original_is_fake = prediction.item() == 0
            original_prediction = "FAKE" if original_is_fake else "REAL"
            
            # Revised confidence threshold logic
            if original_prediction == "FAKE":
                # Require higher confidence for FAKE predictions
                if confidence < 65:  # Increased threshold for FAKE
                    is_fake = False
                    final_prediction = "REAL"
                    confidence_message = f"{confidence:.2f}% (low confidence fake)"
                else:
                    is_fake = True
                    final_prediction = "FAKE"
                    confidence_message = f"{confidence:.2f}%"
            else:
                # More lenient with REAL predictions
                if confidence < 50:  # Lower threshold for REAL
                    is_fake = True
                    final_prediction = "FAKE"
                    confidence_message = f"{confidence:.2f}% (low confidence real)"
                else:
                    is_fake = False
                    final_prediction = "REAL"
                    confidence_message = f"{confidence:.2f}%"
            
            print(f"Original prediction: {original_prediction} with {confidence:.2f}% confidence")
            print(f"Final prediction: {final_prediction} with {confidence_message}")

        response_data = {
            "is_fake": is_fake,
            "confidence": confidence,
            "prediction": final_prediction,
            "original_prediction": original_prediction,
            "confidence_threshold_applied": confidence_message != f"{confidence:.2f}%",
            "frames_used": len(processed_frames),
            "original_frames_count": len(frames),
            "valid_faces_found": len(valid_faces),
            "frame_selection": f"{len(valid_faces)} faces from {max_frames_to_scan} frames",
            "frames_scanned": min(len(frames_to_process), max_frames_to_scan),
            "prediction_basis": f"Result based on {len(valid_faces)} facial frames",
            "sequence_length_used": actual_sequence_length
        }
        print(f"Sending response: {response_data}")
        return response_data

    except Exception as e:
        print(f"Error in predict_frames: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
def test_endpoint():
    return {"status": "success", "message": "API is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)