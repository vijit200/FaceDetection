from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse,JsonResponse
from django.core.files.storage import default_storage
import os
import cv2
import numpy as np
import base64
import json
import pickle
from insightface.app import FaceAnalysis
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import onnxruntime as ort
from .models import UserProfile

# Load Models Once
yolo_model = YOLO("runs/detect/train/weights/best.pt")

session = ort.InferenceSession("modelrgb.onnx")
input_name = session.get_inputs()[0].name
_, _, H, W = session.get_inputs()[0].shape
# shape: (N, D)

face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
face_app.prepare(ctx_id=0)

# === Constants ===
YOLO_INPUT_SIZE = (640, 480)  # Resize for speed
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80

def index(request):
    return render(request, 'index.html')

def capture(request):
   return render(request, 'capture_image.html')

def login_view(request):
    return render(request, 'login.html')

def cards(request):
    name = request.GET.get('name')
    phone = request.GET.get('phone')

    user = UserProfile.objects(username = name ,phone=phone).first()

    if user:
        context = {
            'name': user.username,
            'phone': user.phone,
            'email': user.email,
            'address': user.address,
            'birth_date': user.birth_date,
            'image': user.image  # base64 string
        }
    else:
        context = {
            'name': 'Not Found',
            'phone': 'Not Found',
            'email': 'N/A',
            'address': 'N/A',
            'birth_date': 'N/A',
            'image': ''
        }

    return render(request, 'cards.html', context)

def form_submit(request):
    return render(request, 'Form.html')


def image_to_base64(uploaded_file):
    image_bytes = uploaded_file.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return encoded

@csrf_exempt
def upload(request):
    
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data['image']
        index = data['index']
        person_name = data.get('person_name', 'unknown').strip()

        if not person_name:
            return HttpResponse("Name missing", status=400)

        # Create save directory
        save_dir = os.path.join("dataset", person_name)
        os.makedirs(save_dir, exist_ok=True)

        # Decode base64 image
        encoded = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save image
        filename = os.path.join(save_dir, f"{int(index):03d}.jpg")
        cv2.imwrite(filename, img)

        return HttpResponse(f"[INFO] Saved {filename}")

    return HttpResponse("Only POST allowed", status=405)

@csrf_exempt
def train(request):
    if request.method != 'POST':
        return JsonResponse({"success": False, "message": "Invalid request method"}, status=400)

    try:
        data = json.loads(request.body)
        person_name = data.get("person_name", None)
    except Exception as e:
        return JsonResponse({"success": False, "message": "Invalid JSON", "error": str(e)}, status=400)

    # === Initialize FaceAnalysis from InsightFace ===
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0)

    # === Paths ===
    image_folder = "dataset"
    output_pickle = "face_embeddings.pkl"

    # === Load existing embeddings if available ===
    if os.path.exists(output_pickle):
        with open(output_pickle, 'rb') as f:
            face_embeddings = pickle.load(f)
    else:
        face_embeddings = {}

    # === Only process dataset/person_name if provided ===
    person_folders = [person_name] if person_name else os.listdir(image_folder)

    for person in person_folders:
        person_folder = os.path.join(image_folder, person)
        if not os.path.isdir(person_folder):
            continue

        embeddings = []
        for filename in os.listdir(person_folder):
            filepath = os.path.join(person_folder, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            faces = face_app.get(img)
            if not faces:
                continue

            emb = faces[0].embedding
            embeddings.append(emb)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            face_embeddings[person] = avg_embedding  # Update or add new person

    # === Save updated embeddings ===
    try:
        with open(output_pickle, 'wb') as f:
            pickle.dump(face_embeddings, f)

        shutil.rmtree(image_folder)  # optional: clean up images after training
    except Exception as e:
        return JsonResponse({"success": False, "message": "Failed to save embeddings", "error": str(e)})

    return JsonResponse({
        "success": True,
        "message": "Training completed successfully",
        "embeddings_count": len(face_embeddings)
    })


@csrf_exempt
# === Load models and assets globally ===
def detect_face(request):
    if request.method != 'POST':
        return JsonResponse({"success": False, "message": "Invalid method"})

    try:
        
        with open("face_embeddings.pkl", "rb") as f:
            known_embeddings = pickle.load(f)

        known_names = list(known_embeddings.keys())
        known_vectors = np.array(list(known_embeddings.values()))  
        data = json.loads(request.body)
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        original_h, original_w = frame.shape[:2]

        # === Resize for YOLO inference ===
        resized_frame = cv2.resize(frame, YOLO_INPUT_SIZE)
        scale_x = original_w / YOLO_INPUT_SIZE[0]
        scale_y = original_h / YOLO_INPUT_SIZE[1]

        results = yolo_model(resized_frame, verbose=False)[0]

        name_found = "Unknown"
        draw_color = (0, 0, 255)  # Red by default

        for box in results.boxes.xyxy:
            rx1, ry1, rx2, ry2 = map(int, box)
            # Scale back to original size
            x1, y1 = int(rx1 * scale_x), int(ry1 * scale_y)
            x2, y2 = int(rx2 * scale_x), int(ry2 * scale_y)

            face_width = x2 - x1
            face_height = y2 - y1

            if face_width < MIN_FACE_WIDTH or face_height < MIN_FACE_HEIGHT:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Too Far", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # === Run Liveness Detection ===
            img = cv2.resize(face_crop, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            input_tensor = img[None, :]

            pred = session.run(None, {input_name: input_tensor})[0]
            score = float(pred[0][1]) if pred.shape[-1] == 2 else 1 / (1 + np.exp(-pred[0][0]))
            is_real = score >= 0.5

            if is_real:
                faces = face_app.get(frame)
                for face in faces:
                    fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                    center = ((fx1 + fx2) // 2, (fy1 + fy2) // 2)
                    if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                        similarities = cosine_similarity([face.embedding], known_vectors)[0]
                        max_idx = np.argmax(similarities)
                        if similarities[max_idx] > 0.35:
                            name_found = known_names[max_idx]
                            draw_color = (0, 255, 0)  # Green box for real and matched
                        break

            label = name_found if is_real else "Fake"
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

        # === Encode and return image with box ===
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            "success": name_found != "Unknown",
            "name": name_found,
            "image": f"data:image/jpeg;base64,{encoded_image}"
        })

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})

@csrf_exempt
def save_form(request):
    if request.method == 'POST':
        try:
            photo = request.FILES.get('image')
            username = request.POST.get('username')
            email = request.POST.get('email')
            address = request.POST.get('Address')
            phone = request.POST.get('phone')
            birth_date = request.POST.get('date')  # Format: YYYY-MM-DD


            if UserProfile.objects(username=username, phone=phone).first():
                return JsonResponse({
                    "success": False,
                    "message": "User with this username and phone already exists."
                })

            # Convert uploaded image to base64
            def image_to_base64(image_file):
                return base64.b64encode(image_file.read()).decode('utf-8')

            image_str = image_to_base64(photo)

            # Create and save MongoEngine document
            user = UserProfile(
                image=image_str,
                username=username,
                email=email,
                address=address,
                phone=phone,
                birth_date=birth_date
            )
            user.save()

            return JsonResponse({"success": True})

        except Exception as e:
            return JsonResponse({"success": False, "message": str(e)})

    return JsonResponse({"success": False, "message": "Invalid request method"})


@csrf_exempt
def check_phone(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            phone = data.get('phone')

            if not phone:
                return JsonResponse({'exists': False, 'message': 'Phone number is required.'})

            user = UserProfile.objects(phone=phone).first()
            if user:
                return JsonResponse({'exists': True})
            else:
                return JsonResponse({'exists': False})
        except Exception as e:
            return JsonResponse({'exists': False, 'error': str(e)})
    return JsonResponse({'exists': False, 'message': 'Invalid request method'})

