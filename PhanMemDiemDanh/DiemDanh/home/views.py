import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.shortcuts import render 
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.template import loader
from datetime import datetime
from .models import history_attendance,teachers


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
face_cascade = cv2.CascadeClassifier("home/haarcascade/haarcascade_frontalface_default.xml")

# Dữ liệu sinh viên
student_info = {
    5: {"Họ_tên": "An", "MSSV": "111222", "Giới_tính": "Nam", "Lớp": "ST22A"},
    4: {"Họ_tên": "Phuc", "MSSV": "100000", "Giới_tính": "Nam", "Lớp": "ST22A"}
}

def get_excel_file(student_id):
    return f"diem_danh_{student_id}.xlsx"



# Hàm ghi điểm danh vào Excel
def ghi_diem_danh(student_id):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")  # để tạo tên file duy nhất

    student = student_info.get(student_id, None)
    if not student:
        return "Không xác định sinh viên!"

    # File excel mới cho mỗi lần điểm danh
    excel_file = f"diemdanh_{timestamp_str}.xlsx"

    # Tạo DataFrame và ghi dữ liệu
    df = pd.DataFrame([{
        "MSSV": student["MSSV"],
        "Họ_tên": student["Họ_tên"],
        "Giới_tính": student["Giới_tính"],
        "Lớp": student["Lớp"],
        "Ngày": date_str,
        "Thời_gian": time_str
    }])
    df.to_excel(excel_file, index=False)

    # Lưu DB
    history_attendance.objects.create(
        id=int(datetime.now().timestamp()),
        student_name=student["Họ_tên"],
        student_id=student["MSSV"],
        class_name=student["Lớp"],
        checkin_time=time_str,
        date_attendance=date_str,
        status="present"
    )

    return f"✔ Điểm danh: {student['Họ_tên']} ({student['MSSV']}) lúc {time_str} → File: {excel_file}"

    
    return f"{student['Họ_tên']} đã điểm danh hôm nay!"


# Hàm nhận diện khuôn mặt từ camera
import cv2

import time
import threading

def ghi_diem_danh_async(student_id):
    """Ghi điểm danh trong thread riêng"""
    threading.Thread(target=ghi_diem_danh, args=(student_id,)).start()


def detect_faces():
    cam = cv2.VideoCapture(0)
    try:
        cam.set(3, 640)
        cam.set(4, 480)

        detected = False
        start_time = None

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                id, confidence = recognizer.predict(face_roi)

                if confidence < 70:
                    student = student_info.get(id, {"Họ_tên": "No"})
                    name = student["Họ_tên"]
                    color = (0, 255, 0)

                    if not detected:
                        # lần đầu ghi điểm danh
                        ghi_diem_danh_async(id)
                        detected = True
                        start_time = time.time()
                else:
                    name = "No"
                    color = (0, 0, 255)

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            _, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # nếu đã nhận diện và quá 10s thì tắt camera
            if detected and (time.time() - start_time > 30):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()



    cam = cv2.VideoCapture(0)  # Mở camera

    try:
        cam.set(3, 640)  # Chiều rộng khung hình
        cam.set(4, 480)  # Chiều cao khung hình

        detected = False  # Biến để kiểm tra xem đã nhận diện lần đầu chưa
        start_time = None 
        
        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                id, confidence = recognizer.predict(face_roi)

                if confidence < 70 and not detected:
                    result = ghi_diem_danh(id)  # Ghi nhận điểm danh
                    student = student_info.get(id, {"Họ_tên": "No"})
                    name = student["Họ_tên"]
                    color = (0, 255, 0)
                    detected = True  # Đánh dấu đã nhận diện lần đầu
                    start_time = time.time()

                else:
                    name = "No"
                    color = (0, 0, 255)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            _, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if detected:
                break  # Thoát vòng lặp sau khi nhận diện lần đầu

    finally:
        cam.release()
        cv2.destroyAllWindows()


@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(detect_faces(), content_type='multipart/x-mixed-replace; boundary=frame')
# dat ten get_ cho no dong bo voi cac ham khac
def get_login(request):
    if request.method == 'POST':
        username = request.POST.get('un')
        password = request.POST.get('pw')
        user = authenticate(request, username=username, password=password)
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)

                if user.is_superuser:
                    return redirect('/admin/')
                elif hasattr(user, 'userprofile') and user.userprofile.is_teacher:
                    return redirect('/courses/')
                else:
                    next_url = request.GET.get('course', 'home') 
                    return redirect(next_url)
            else:
                messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng')
        else:
            messages.error(request, 'Vui lòng nhập đầy đủ thông tin')

    return render(request, 'home/loginPage.html')

@login_required(login_url='home/loginPage.html')
def get_home(request):
    today = datetime.today()
    context = {
        'current_month': today.strftime("%B %Y"),
        'current_day': today.day,
    }
    return render(request, 'home/home.html', context)

def get_profile(request):
    teacher = teachers.objects.all()  # Lấy toàn bộ danh sách giáo viên
    return render(request, 'home/profile.html', {'teacher': teacher})
    
def get_profileEdit(request):
    return render(request, 'home/profile-edit.html')

def get_history(request):
    user = request.user

    if user.groups.filter(name="Teachers").exists():
        # Nếu là giáo viên, xem tất cả lịch sử điểm danh
        records = history_attendance.objects.all()
    else:
        # Nếu là học sinh, chỉ xem lịch sử điểm danh của chính mình
        records = history_attendance.objects.filter(student_id=user.username)

    return render(request, 'home/history.html', {'records': records})


def get_face_recognition(request):
    return render(request, "home/face_recognition.html")  # Load giao diện nhận diện

    