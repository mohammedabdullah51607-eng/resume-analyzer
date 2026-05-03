from flask import Flask, request, jsonify, render_template, send_file
import pdfplumber, docx, re, pickle, csv
from pymongo import MongoClient
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import os

# ===== NLTK FIX (IMPORTANT FOR RENDER) =====
import nltk
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

app = Flask(__name__)

# ===== LOAD MODELS =====
models = pickle.load(open("models.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# ===== DATABASE (SAFE FOR DEPLOYMENT) =====
USE_DB = False  # 🔴 Set False for Render (no MongoDB)

if USE_DB:
    client = MongoClient("mongodb://localhost:27017/")
    db = client["resume_db"]
    collection = db["candidates"]
else:
    collection = None

# ===== ROUTES =====
@app.route("/")
def user():
    return render_template("user.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# ===== CLEAR DATA =====
@app.route("/clear", methods=["POST"])
def clear():
    if collection:
        res = collection.delete_many({})
        return jsonify({"message":"Data cleared","deleted":res.deleted_count})
    return jsonify({"message":"DB disabled"})

# ===== CLEAN TEXT =====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n',' ',text)
    text = re.sub(r'[^a-zA-Z ]',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

# ===== EXTRACT TEXT =====
def extract_text(file):
    stream = BytesIO(file.read())

    if file.filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(stream) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text += t
        return text

    elif file.filename.endswith(".docx"):
        doc = docx.Document(stream)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise Exception("Unsupported file")

# ===== SKILLS =====
def get_skills(text):
    skills = [
        "python","java","sql","machine learning",
        "html","css","javascript","react","flask","mongodb"
    ]
    return [s.title() for s in skills if s in text.lower()]

# ===== EMAIL =====
def get_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not Found"

# ===== ROLE PREDICTION =====
def predict_role(text):
    vec = vectorizer.transform([clean_text(text)])
    scores = {}

    for model in models.values():
        if hasattr(model,"predict_proba"):
            probs = model.predict_proba(vec)[0]
        else:
            raw = model.decision_function(vec)[0]
            probs = (raw - raw.min())/(raw.max()-raw.min()+1e-6)

        for i, role in enumerate(model.classes_):
            scores[role] = scores.get(role,0) + probs[i]

    total = sum(scores.values()) + 1e-6
    scores = {k:(v/total)*100 for k,v in scores.items()}

    if not scores:
        return "Unknown", 0

    best = max(scores, key=scores.get)
    return best, round(scores.get(best, 0), 2)

# ===== MATCH SCORE =====
def match_score(resume, job):
    vec = vectorizer.transform([resume, job])
    score = cosine_similarity(vec[0], vec[1])[0][0]

    if score != score:
        score = 0

    return round(score * 100, 2)

# ===== FINAL SCORE =====
def final_score(conf, match, skills):
    return round(conf*0.5 + match*0.3 + len(skills)*5*0.2, 2)

# ===== DOMAIN GRAPH =====
def get_domain_scores(conf, match):
    try:
        conf = float(conf) if conf else 0
        match = float(match) if match else 0
    except:
        conf, match = 0, 0

    return {
        "Web Dev": round(max(5, min(100, conf * 1.2)), 2),
        "Data Science": round(max(5, min(100, match * 1.1)), 2),
        "Java Dev": round(max(5, min(100, (conf + match) / 2)), 2)
    }

# ===== USER ANALYSIS =====
@app.route("/user_upload", methods=["POST"])
def user_upload():
    try:
        file = request.files["resume"]
        job = request.form.get("job_description","")

        text = extract_text(file)

        skills = get_skills(text)
        role, conf = predict_role(text)
        match = match_score(text, job)
        score = final_score(conf, match, skills)

        job_words = job.lower().split()
        missing = list(set([w for w in job_words if w not in text.lower()]))[:5]

        domains = get_domain_scores(conf, match)

        suggestion = (
            "Excellent profile!" if score > 75 else
            "Good profile, improve skills." if score > 50 else
            "Needs improvement."
        )

        return jsonify({
            "email": get_email(text),
            "role": role,
            "confidence": conf,
            "match": match,
            "skills": skills,
            "final_score": score,
            "missing_skills": missing,
            "suggestion": suggestion,
            "domains": domains
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ===== BULK UPLOAD =====
@app.route("/bulk_upload", methods=["POST"])
def bulk_upload():
    results = []

    files = request.files.getlist("resumes")
    job = request.form.get("job_description","")

    for f in files:
        try:
            text = extract_text(f)

            skills = get_skills(text)
            role, conf = predict_role(text)
            match = match_score(text, job)
            score = final_score(conf, match, skills)

            domains = get_domain_scores(conf, match)

            data = {
                "email": get_email(text),
                "role": role,
                "confidence": conf,
                "skills": skills,
                "match_score": match,
                "final_score": score,
                "domains": domains
            }

            if collection:
                collection.insert_one(data)

            results.append(data)

        except Exception as e:
            print("ERROR:", e)

    results = sorted(results, key=lambda x:x["final_score"], reverse=True)
    return jsonify({"results": results})

# ===== SHORTLIST =====
@app.route("/shortlist")
def shortlist():
    if collection:
        data = list(collection.find({}, {"_id":0}))
    else:
        data = []

    data = sorted(data, key=lambda x:x.get("final_score",0), reverse=True)
    return jsonify({"shortlisted": data[:5]})

# ===== DOWNLOAD =====
@app.route("/download")
def download():
    if collection:
        data = list(collection.find({}, {"_id":0}))
    else:
        return jsonify({"error": "DB disabled"})

    file = "results.csv"

    with open(file,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    return send_file(file, as_attachment=True)

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)