from flask import Flask, request, session, redirect, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import sqlite3, random, time, os, requests, hashlib, hmac
import torch, cv2, numpy as np
from PIL import Image
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

# ─── RATE LIMITER ───
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── CONFIG FROM ENV ───
FAST2SMS_KEY   = os.environ.get("FAST2SMS_KEY", "")
OWNER_UID      = os.environ.get("OWNER_UID", "")
OWNER_PWD_HASH = os.environ.get("OWNER_PWD_HASH", "")  # sha256 hex of password
DAILY_LIMIT    = int(os.environ.get("DAILY_LIMIT", 10))

# ─── DATABASE ───
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        phone       TEXT PRIMARY KEY,
        name        TEXT,
        email       TEXT,
        otp         TEXT,
        otp_time    REAL,
        otp_attempts INTEGER DEFAULT 0,
        count       INTEGER DEFAULT 0,
        last_reset  REAL DEFAULT 0
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS premium_users (
        phone       TEXT PRIMARY KEY,
        plan        TEXT,
        activated   REAL,
        expires     REAL
    )''')
    conn.commit()
    conn.close()

init_db()

# ─── LOAD MiDaS ───
print("Loading MiDaS model...")
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
print("MiDaS loaded ✓")

# ─── HELPERS ───
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "phone" not in session:
            return jsonify({"status": "error", "message": "Login required"}), 401
        return f(*args, **kwargs)
    return decorated

def check_daily_reset(phone):
    """Reset daily count if 24h passed."""
    conn = get_db()
    row = conn.execute("SELECT count, last_reset FROM users WHERE phone=?", (phone,)).fetchone()
    if row and (time.time() - row["last_reset"]) > 86400:
        conn.execute("UPDATE users SET count=0, last_reset=? WHERE phone=?", (time.time(), phone))
        conn.commit()
    conn.close()

def get_remaining(phone):
    check_daily_reset(phone)
    conn = get_db()
    row = conn.execute("SELECT count FROM users WHERE phone=?", (phone,)).fetchone()
    conn.close()
    # Check premium
    is_premium = check_premium(phone)
    if is_premium:
        return 999  # unlimited
    return max(0, DAILY_LIMIT - (row["count"] if row else 0))

def check_premium(phone):
    conn = get_db()
    row = conn.execute(
        "SELECT expires FROM premium_users WHERE phone=? AND expires > ?",
        (phone, time.time())
    ).fetchone()
    conn.close()
    return row is not None

def send_sms_fast2sms(phone, otp):
    """Send OTP via Fast2SMS."""
    try:
        url = "https://www.fast2sms.com/dev/bulkV2"
        params = {
            "authorization": FAST2SMS_KEY,
            "variables_values": otp,
            "route": "otp",
            "numbers": phone,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return data.get("return", False)
    except Exception as e:
        print(f"SMS Error: {e}")
        return False

def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

# ─── SERVE FRONTEND ───
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# ─── OTP SEND ───
@app.route("/send_otp", methods=["POST"])
@limiter.limit("5 per minute")
def send_otp():
    phone = request.form.get("phone", "").strip()

    # Validate phone (10 digits, Indian)
    digits = "".join(filter(str.isdigit, phone))
    if len(digits) < 10:
        return jsonify({"status": "error", "message": "Invalid phone number"}), 400

    phone = digits[-10:]  # Take last 10 digits

    otp = str(random.randint(1000, 9999))
    otp_time = time.time()

    conn = get_db()
    conn.execute("""INSERT INTO users (phone, otp, otp_time, otp_attempts, count, last_reset)
                    VALUES (?,?,?,0,0,?)
                    ON CONFLICT(phone) DO UPDATE SET
                    otp=excluded.otp,
                    otp_time=excluded.otp_time,
                    otp_attempts=0""",
                 (phone, otp, otp_time, time.time()))
    conn.commit()
    conn.close()

    if FAST2SMS_KEY:
        success = send_sms_fast2sms(phone, otp)
        if not success:
            print(f"[FALLBACK] OTP for {phone}: {otp}")
    else:
        # Dev mode - print to console
        print(f"\n{'='*40}")
        print(f"  OTP for {phone}: {otp}")
        print(f"{'='*40}\n")

    return jsonify({"status": "sent"})

# ─── OTP VERIFY ───
@app.route("/verify_otp", methods=["POST"])
@limiter.limit("10 per minute")
def verify_otp():
    phone = request.form.get("phone", "").strip()
    entered = request.form.get("otp", "").strip()

    digits = "".join(filter(str.isdigit, phone))
    phone = digits[-10:]

    conn = get_db()
    row = conn.execute(
        "SELECT otp, otp_time, otp_attempts FROM users WHERE phone=?", (phone,)
    ).fetchone()

    if not row:
        conn.close()
        return jsonify({"status": "fail", "message": "Phone not found"}), 400

    # Check attempts (max 5)
    if row["otp_attempts"] >= 5:
        conn.close()
        return jsonify({"status": "fail", "message": "Too many attempts. Request new OTP."}), 429

    # Check expiry (5 minutes)
    if time.time() - row["otp_time"] > 300:
        conn.close()
        return jsonify({"status": "fail", "message": "OTP expired. Request a new one."}), 400

    # Increment attempts
    conn.execute("UPDATE users SET otp_attempts=otp_attempts+1 WHERE phone=?", (phone,))
    conn.commit()

    if row["otp"] == entered:
        # Clear OTP after success
        conn.execute("UPDATE users SET otp=NULL, otp_time=NULL WHERE phone=?", (phone,))
        conn.commit()
        conn.close()
        session["phone"] = phone
        session.permanent = True
        return jsonify({"status": "ok"})

    conn.close()
    return jsonify({"status": "fail", "message": "Wrong OTP"}), 400

# ─── SAVE PROFILE ───
@app.route("/save_profile", methods=["POST"])
@login_required
def save_profile():
    name  = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()

    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400

    conn = get_db()
    conn.execute("UPDATE users SET name=?, email=? WHERE phone=?",
                 (name, email, session["phone"]))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "name": name})

# ─── GET USER INFO ───
@app.route("/me")
@login_required
def me():
    conn = get_db()
    row = conn.execute(
        "SELECT name, email, count FROM users WHERE phone=?", (session["phone"],)
    ).fetchone()
    conn.close()

    remaining = get_remaining(session["phone"])
    is_premium = check_premium(session["phone"])

    return jsonify({
        "status": "ok",
        "phone": session["phone"],
        "name": row["name"] if row else None,
        "email": row["email"] if row else None,
        "count": row["count"] if row else 0,
        "remaining": remaining,
        "is_premium": is_premium,
        "daily_limit": DAILY_LIMIT
    })

# ─── GENERATE 3D ───
@app.route("/generate_3d", methods=["POST"])
@login_required
def generate_3d():
    phone = session["phone"]

    # Check remaining
    remaining = get_remaining(phone)
    if remaining <= 0:
        return jsonify({"status": "error", "message": "Daily limit reached. Upgrade to Premium!"}), 429

    # File validation
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type. Use PNG, JPG, WEBP."}), 400

    # Save securely
    filename = secure_filename(file.filename)
    # Add timestamp to avoid collisions
    base, ext = os.path.splitext(filename)
    filename = f"{base}_{int(time.time())}{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        # Run MiDaS
        img = cv2.imread(path)
        if img is None:
            return jsonify({"status": "error", "message": "Could not read image"}), 400

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        input_tensor = midas_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            prediction = midas(input_tensor)
            depth = prediction.squeeze().cpu().numpy()

        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        depth_filename = f"{base}_{int(time.time())}_depth.png"
        depth_path = os.path.join(UPLOAD_FOLDER, depth_filename)
        cv2.imwrite(depth_path, depth_norm)

        # Increment count AFTER success
        conn = get_db()
        conn.execute("UPDATE users SET count=count+1 WHERE phone=?", (phone,))
        conn.commit()
        conn.close()

        return jsonify({
            "status": "ok",
            "depth": depth_path,
            "remaining": remaining - 1
        })

    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({"status": "error", "message": "Processing failed. Try again."}), 500
    finally:
        # Clean up original upload
        if os.path.exists(path):
            os.remove(path)

# ─── LOGOUT ───
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "ok"})

# ─── PREMIUM ACTIVATE (Owner only via API) ───
@app.route("/activate_premium", methods=["POST"])
def activate_premium():
    uid  = request.form.get("uid", "")
    pwd  = request.form.get("pwd", "")
    phone = request.form.get("phone", "").strip()
    plan  = request.form.get("plan", "monthly")  # monthly / yearly

    if uid != OWNER_UID or hash_pwd(pwd) != OWNER_PWD_HASH:
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    duration = 30 * 86400 if plan == "monthly" else 365 * 86400
    expires = time.time() + duration

    conn = get_db()
    conn.execute("""INSERT INTO premium_users (phone, plan, activated, expires)
                    VALUES (?,?,?,?)
                    ON CONFLICT(phone) DO UPDATE SET plan=excluded.plan,
                    activated=excluded.activated, expires=excluded.expires""",
                 (phone, plan, time.time(), expires))
    conn.commit()
    conn.close()

    return jsonify({"status": "ok", "message": f"Premium activated for {phone} ({plan})"})

# ─── OWNER DASHBOARD ───
@app.route("/owner_stats", methods=["POST"])
@limiter.limit("20 per minute")
def owner_stats():
    uid = request.form.get("uid", "")
    pwd = request.form.get("pwd", "")

    if uid != OWNER_UID or hash_pwd(pwd) != OWNER_PWD_HASH:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 403

    conn = get_db()
    total_users    = conn.execute("SELECT COUNT(*) as c FROM users WHERE name IS NOT NULL").fetchone()["c"]
    total_premium  = conn.execute("SELECT COUNT(*) as c FROM premium_users WHERE expires > ?", (time.time(),)).fetchone()["c"]
    total_gen_today = conn.execute(
        "SELECT SUM(count) as s FROM users WHERE last_reset > ?", (time.time() - 86400,)
    ).fetchone()["s"] or 0
    conn.close()

    return jsonify({
        "status": "ok",
        "total_users": total_users,
        "premium_users": total_premium,
        "generations_today": total_gen_today
    })

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=5000)
