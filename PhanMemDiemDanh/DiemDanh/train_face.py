import cv2
import os
import numpy as np
from PIL import Image

# ==============================
# 1. Cấu hình
# ==============================
student_id = 5   # Đổi ID cho bạn (VD: 5)
student_name = "TestUser"  # Tên của bạn
dataset_path = "dataset"
trainer_path = "trainer"
face_cascade_path = "home/haarcascade/haarcascade_frontalface_default.xml"

# ==============================
# 2. Thu thập dữ liệu
# ==============================
def collect_faces():
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("📸 Đang chụp ảnh khuôn mặt. Nhìn vào camera...")
    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{dataset_path}/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow('Collecting Faces', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:   # ESC để thoát
            break
        elif count >= 50:  
            break

    print(f"✅ Hoàn tất thu thập {count} ảnh")
    cam.release()
    cv2.destroyAllWindows()


# ==============================
# 3. Huấn luyện / Cập nhật dữ liệu
# ==============================
def train_or_update_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(face_cascade_path)

    def get_images_and_labels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        return faceSamples, ids

    print("⚙️ Đang huấn luyện/cập nhật dữ liệu...")
    faces, ids = get_images_and_labels(dataset_path)

    if not os.path.exists(trainer_path):
        os.makedirs(trainer_path)

    trainer_file = f"{trainer_path}/trainer.yml"

    if os.path.exists(trainer_file):
        # Đọc dữ liệu cũ và update
        recognizer.read(trainer_file)
        recognizer.update(faces, np.array(ids))
        print("🔄 Đã cập nhật thêm khuôn mặt mới")
    else:
        # Huấn luyện mới hoàn toàn
        recognizer.train(faces, np.array(ids))
        print("🆕 Huấn luyện model lần đầu")

    recognizer.write(trainer_file)
    print(f"✅ Lưu trainer vào {trainer_file}")


# ==============================
# 4. Chạy chương trình
# ==============================
if __name__ == "__main__":
    collect_faces()
    train_or_update_faces()
    print("🎉 Sẵn sàng nhận diện!")
