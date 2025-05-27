from flask import Flask, render_template, request, session, Response, jsonify
import cv2
import numpy as np
import os
from datetime import datetime, date
import sqlite3
import json
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from models import EmbeddingModel, load_model, get_embedding
from settings import IMG_PER_USER, MODEL_PATH, DEVICE, IMG_SIZE, THRESHOLD, WIN_SIZE, MIN_VOTES
from utils import build_db, get_db_info, remove_db
from collections import deque, Counter
from face_func import add_face, remove_face, search_face
import queue
import threading

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Initialize database and create table if not exists
def init_db():
    conn = sqlite3.connect('information.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS Attendance
                    (NAME TEXT NOT NULL,
                     Time TEXT NOT NULL,
                     Date TEXT NOT NULL)''')
    conn.commit()
    conn.close()

# Call init_db when app starts
init_db()

# Khởi tạo các model và index
embedder = load_model(EmbeddingModel(), MODEL_PATH, DEVICE)
index, info = build_db(embedder)

# Global variables for face recognition
detector = MTCNN(image_size=IMG_SIZE, margin=10, keep_all=False, post_process=False, device=DEVICE)
votes = deque(maxlen=WIN_SIZE)
recognized_names = set()

# Queue for attendance events
attendance_queue = queue.Queue()

# Global variables for face registration
registration_queue = queue.Queue()
current_registration = {
    'pid': None,
    'name': None,
    'count': 0
}
current_mode = 'attendance'  # 'attendance' or 'registration'

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            bbox, _ = detector.detect(frame)
            draw_frame = frame.copy()
            
            if bbox is not None and len(bbox) == 1:
                x1, y1, x2, y2 = [int(v) for v in bbox[0]]
                
                if current_mode == 'registration':
                    # Registration mode - only detect face
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(draw_frame, f"Face Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Check for registration capture request
                    try:
                        event = registration_queue.get_nowait()
                        if event['type'] == 'capture':
                            face_img = detector.extract(
                                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 
                                bbox,
                                save_path=os.path.join('gallery/user', f"{event['pid']}_{event['name']}", f"{event['pid']}_{event['name']}_{current_registration['count']:04d}.jpg")
                            )
                            if face_img is not None:
                                # Add embedding to index
                                face_emb = get_embedding(embedder, face_img)
                                index.add(face_emb)
                                info.append({
                                    'pid': event['pid'],
                                    'name': event['name'],
                                    'img_path': os.path.join('gallery/user', f"{event['pid']}_{event['name']}", f"{event['pid']}_{event['name']}_{current_registration['count']:04d}.jpg")
                                })
                                current_registration['count'] += 1
                                # Save index and info when all images are captured
                                if current_registration['count'] >= IMG_PER_USER:
                                    save_index(index)
                                    save_info(info)
                    except queue.Empty:
                        pass
                else:
                    # Attendance mode - recognize face
                    face_img = detector.extract(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), bbox, save_path=None)
                    if face_img is not None:
                        face_emb = get_embedding(embedder, face_img)
                        D, I = index.search(face_emb, 1)
                        score, idx = D[0][0], I[0][0]

                        if score >= THRESHOLD:
                            name = info[idx]['name']
                            pid = info[idx]['pid']
                            votes.append((name, score, pid))
                        else:
                            votes.append(('Unknown', score, None))
                        
                        # Voting phase
                        label, color = 'Unknown', (0, 0, 255)
                        if len(votes) == WIN_SIZE:
                            names = [(n, p) for n, s, p in votes if n != 'Unknown']
                            if names:
                                most_common, cnt = Counter([n[0] for n in names]).most_common(1)[0]
                                if cnt >= MIN_VOTES:
                                    avg_score = np.mean([s for n, s, p in votes if n == most_common])
                                    label = f'{most_common} ({avg_score:.2f})'
                                    color = (0, 255, 0)
                                    
                                    # Save attendance data if not already saved in this session
                                    if most_common not in recognized_names:
                                        markData(most_common)
                                        recognized_names.add(most_common)
                                        # Get the corresponding pid
                                        pid = next(p for n, s, p in votes if n == most_common)
                                        # Put attendance event in queue
                                        attendance_queue.put({
                                            'type': 'attendance_marked',
                                            'name': most_common,
                                            'id': pid
                                        })
                        
                        # Rendering phase
                        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(draw_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            ret, buffer = cv2.imencode('.jpg', draw_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_events')
def attendance_events():
    def generate():
        while True:
            try:
                # Get event from queue with timeout
                event = attendance_queue.get(timeout=1)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send empty event to keep connection alive
                yield "data: {}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/recognize', methods=["GET", "POST"])
def recognize():
    if request.method == "POST":
        # Reset recognized names for new session
        global recognized_names, current_mode
        recognized_names = set()
        current_mode = 'attendance'
        return render_template('recognize.html')
    return render_template('main.html')

@app.route('/exit_recognition', methods=["POST"])
def exit_recognition():
    # Reset recognized names when exiting recognition mode
    global recognized_names
    recognized_names = set()
    return jsonify({'status': 'success'})

@app.route('/')
def index_route():  # Đổi tên hàm để tránh xung đột với biến index
    return render_template('main.html')

@app.route('/new', methods=['GET', 'POST'])
def new():
    if 'username' not in session:
        return render_template('form.html')
    if request.method == "POST":
        return render_template('index.html')
    return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if 'username' not in session:
        return render_template('form.html')
    if request.method == "POST":
        name1 = request.form['name1']
        name2 = request.form['name2']
        # Check if ID already exists
        if name1 in [i['pid'] for i in info]:
            return jsonify({'error': 'ID already exists'}), 400
        # Initialize registration session
        global current_registration, current_mode
        current_registration = {
            'pid': name1,
            'name': name2,
            'count': 0
        }
        current_mode = 'registration'
        return render_template('register.html', pid=name1, name=name2, img_per_user=IMG_PER_USER)
    return 'All is not well'

@app.route('/capture_face', methods=['POST'])
def capture_face():
    if 'username' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    pid = data.get('pid')
    name = data.get('name')
    
    if not pid or not name:
        return jsonify({'success': False, 'error': 'Missing pid or name'}), 400
    
    # Put capture request in queue
    registration_queue.put({
        'type': 'capture',
        'pid': pid,
        'name': name
    })
    
    return jsonify({'success': True})

@app.route('/remove_face', methods=['POST'])
def remove_face_route():
    if 'username' not in session:
        return "Unauthorized", 401
    if request.method == "POST":
        try:
            global info, index
            pid = request.form['pid']
            # Get name from info before removing face
            name = None
            for face in info:
                if face['pid'] == pid:
                    name = face['name']
                    break
            
            if name:
                # Remove face from face recognition system
                remove_face(pid, index, info, embedder)
                
                # Remove attendance records from database
                conn = sqlite3.connect('information.db')
                conn.execute("DELETE FROM Attendance WHERE NAME = ?", (name,))
                conn.commit()
                conn.close()
                
                # Update global info variable
                info = [i for i in info if i['pid'] != pid]
                
                # Rebuild the index to match the updated info
                index, info = build_db(embedder)
                
                return "Face and attendance records removed successfully"
            else:
                return "Face not found", 404
        except Exception as e:
            return f"Error: {str(e)}", 500
    return "Invalid request"

def markAttendance(name):
    with open('attendance.csv', 'r+', errors='ignore') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            f.writelines(f'\n{name},{dtString}')

def markData(name):
    now = datetime.now()
    dtString = now.strftime('%H:%M')
    today = date.today()
    
    conn = sqlite3.connect('information.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS Attendance
                    (NAME TEXT NOT NULL,
                     Time TEXT NOT NULL,
                     Date TEXT NOT NULL)''')
    
    conn.execute("INSERT or Ignore into Attendance (NAME,Time,Date) values (?,?,?)",
                (name, dtString, today))
    conn.commit()
    conn.close()

@app.route('/login', methods=['POST'])
def login():
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    password = json_data['password']
    
    df = pd.read_csv('cred.csv')
    if len(df.loc[df['username'] == username]['password'].values) > 0:
        if df.loc[df['username'] == username]['password'].values[0] == password:
            session['username'] = username
            return json.dumps({'status': 'success', 'redirect': '/'})
    return json.dumps({'status': 'failed'})

@app.route('/checklogin')
def checklogin():
    if 'username' in session:
        return session['username']
    return 'False'

@app.route('/logout')
def logout():
    session.pop('username', None)
    return 'success'

@app.route('/how', methods=["GET", "POST"])
def how():
    return render_template('form.html')

@app.route('/data', methods=["GET", "POST"])
def data():
    if request.method == "POST":
        today = date.today()
        conn = sqlite3.connect('information.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance where Date=?", (today,))
        rows = cur.fetchall()
        conn.close()
        return render_template('form2.html', rows=rows)
    return render_template('form1.html')

@app.route('/whole', methods=["GET", "POST"])
def whole():
    today = date.today()
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance")
    rows = cur.fetchall()
    return render_template('form3.html', rows=rows)

@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashboard_data')
def dashboard_data():
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get total students count
    total_students_query = """
    SELECT COUNT(DISTINCT NAME) as total 
    FROM Attendance
    """
    total_students = cur.execute(total_students_query).fetchone()['total']
    
    # Get today's attendance
    today = date.today()
    today_query = """
    SELECT COUNT(DISTINCT NAME) as count 
    FROM Attendance 
    WHERE Date = ?
    """
    today_attendance = cur.execute(today_query, (today,)).fetchone()['count']
    
    # Get daily attendance counts
    daily_query = """
    SELECT Date, COUNT(DISTINCT NAME) as count 
    FROM Attendance 
    GROUP BY Date 
    ORDER BY Date
    """
    daily_data = cur.execute(daily_query).fetchall()
    
    # Get time distribution
    time_query = """
    SELECT substr(Time, 1, 2) as hour, COUNT(*) as count 
    FROM Attendance 
    GROUP BY hour 
    ORDER BY hour
    """
    time_data = cur.execute(time_query).fetchall()
    
    # Get attendance trend (last 30 days)
    trend_query = """
    SELECT Date, COUNT(DISTINCT NAME) as count 
    FROM Attendance 
    WHERE Date >= date('now', '-30 days')
    GROUP BY Date 
    ORDER BY Date
    """
    trend_data = cur.execute(trend_query).fetchall()
    
    # Get weekly attendance
    weekly_query = """
    SELECT strftime('%Y-%W', Date) as week,
           COUNT(DISTINCT NAME) as count
    FROM Attendance
    GROUP BY week
    ORDER BY week DESC
    LIMIT 12
    """
    weekly_data = cur.execute(weekly_query).fetchall()
    
    # Get monthly attendance
    monthly_query = """
    SELECT strftime('%Y-%m', Date) as month,
           COUNT(DISTINCT NAME) as count
    FROM Attendance
    GROUP BY month
    ORDER BY month DESC
    LIMIT 12
    """
    monthly_data = cur.execute(monthly_query).fetchall()
    
    # Get student attendance statistics
    student_stats_query = """
    SELECT NAME,
           COUNT(*) as total_attendance,
           MIN(Date) as first_attendance,
           MAX(Date) as last_attendance,
           COUNT(DISTINCT Date) as days_present
    FROM Attendance
    GROUP BY NAME
    ORDER BY total_attendance DESC
    """
    student_stats = cur.execute(student_stats_query).fetchall()
    
    # Get attendance rate by day of week
    day_of_week_query = """
    SELECT strftime('%w', Date) as day,
           COUNT(DISTINCT NAME) as count
    FROM Attendance
    GROUP BY day
    ORDER BY day
    """
    day_of_week_data = cur.execute(day_of_week_query).fetchall()
    
    conn.close()
    
    # Format data for JSON response
    response = {
        'summary': {
            'total_students': total_students,
            'today_attendance': today_attendance,
            'attendance_rate': round((today_attendance / total_students * 100) if total_students > 0 else 0, 2)
        },
        'daily_attendance': {
            'dates': [row['Date'] for row in daily_data],
            'counts': [row['count'] for row in daily_data]
        },
        'time_distribution': {
            'times': [f"{row['hour']}:00" for row in time_data],
            'counts': [row['count'] for row in time_data]
        },
        'attendance_trend': {
            'dates': [row['Date'] for row in trend_data],
            'counts': [row['count'] for row in trend_data]
        },
        'weekly_attendance': {
            'weeks': [row['week'] for row in weekly_data],
            'counts': [row['count'] for row in weekly_data]
        },
        'monthly_attendance': {
            'months': [row['month'] for row in monthly_data],
            'counts': [row['count'] for row in monthly_data]
        },
        'student_stats': [{
            'name': row['NAME'],
            'total_attendance': row['total_attendance'],
            'first_attendance': row['first_attendance'],
            'last_attendance': row['last_attendance'],
            'days_present': row['days_present']
        } for row in student_stats],
        'day_of_week': {
            'days': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
            'counts': [row['count'] for row in day_of_week_data]
        }
    }
    
    return json.dumps(response)

@app.route('/dashboard_filtered_data')
def dashboard_filtered_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return json.dumps({'error': 'Start date and end date are required'}), 400
    
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get filtered daily attendance
    daily_query = """
    SELECT Date, COUNT(DISTINCT NAME) as count 
    FROM Attendance 
    WHERE Date BETWEEN ? AND ?
    GROUP BY Date 
    ORDER BY Date
    """
    daily_data = cur.execute(daily_query, (start_date, end_date)).fetchall()
    
    # Get filtered student stats
    student_stats_query = """
    SELECT NAME,
           COUNT(*) as total_attendance,
           MIN(Date) as first_attendance,
           MAX(Date) as last_attendance,
           COUNT(DISTINCT Date) as days_present
    FROM Attendance
    WHERE Date BETWEEN ? AND ?
    GROUP BY NAME
    ORDER BY total_attendance DESC
    """
    student_stats = cur.execute(student_stats_query, (start_date, end_date)).fetchall()
    
    conn.close()
    
    response = {
        'daily_attendance': {
            'dates': [row['Date'] for row in daily_data],
            'counts': [row['count'] for row in daily_data]
        },
        'student_stats': [{
            'name': row['NAME'],
            'total_attendance': row['total_attendance'],
            'first_attendance': row['first_attendance'],
            'last_attendance': row['last_attendance'],
            'days_present': row['days_present']
        } for row in student_stats]
    }
    
    return json.dumps(response)

@app.route('/registered_faces')
def registered_faces():
    if 'username' not in session:
        return render_template('form.html')
    
    # Get unique faces from info
    unique_faces = []
    seen_pids = set()
    for face in info:
        if face['pid'] not in seen_pids:
            unique_faces.append({
                'pid': face['pid'],
                'name': face['name']
            })
            seen_pids.add(face['pid'])
    
    return render_template('registered_faces.html', faces=unique_faces)

if __name__ == '__main__':
    app.run(debug=True)