from flask import Flask, render_template, request, redirect, url_for, session,jsonify
from flask_sqlalchemy import SQLAlchemy
import nltk 
import yake
import requests
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from summa.summarizer import summarize
from flask import session
from flask import send_file
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure SQLAlchemy to use Oracle database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://tejas:tejas@localhost:3306/student'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)
app.secret_key = 'your_secret_key'

# Define the User model for SQLAlchemy
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)


#exam model
class exam_scores(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    score = db.Column(db.Float)

# Define the MCQ model
class MCQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(255), nullable=False)  # New column for test name
    question = db.Column(db.Text, nullable=False)
    choice1 = db.Column(db.String(255), nullable=False)
    choice2 = db.Column(db.String(255), nullable=False)
    choice3 = db.Column(db.String(255), nullable=False)
    choice4 = db.Column(db.String(255), nullable=False)
    correct_answer = db.Column(db.String(255), nullable=False)


# Helper functions for text processing and MCQ generation

def read_text_with_encoding(filename, encodings):
    for encoding in encodings:
        try:
            with open(filename, encoding=encoding) as file:
                full_text = file.read()
                return full_text
        except UnicodeDecodeError:
            pass
    raise Exception("Failed to read the file with the provided encodings")

def get_nouns_multipartite(text):
    out = []
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=20, features=None)
    keyphrases = kw_extractor.extract_keywords(text)
    for key in keyphrases:
        out.append(key[0])
    return out

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)
    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent, word):
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def get_distractors_conceptnet(word):
    word = word.lower()
    original_word = word
    if (len(word.split()) > 0):
        word = word.replace(" ", "_")
    distractor_list = []
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
    return distractor_list




# Route for generating MCQs

@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    test_name = request.form.get('test_name')  # Retrieve test name from the form
    text_input = request.form.get('text_input')  # Get text input from the form
    file = request.files.get('file')  # Get the uploaded file from the form

    if text_input:  # If text input is provided
        full_text = text_input
    elif file:  # If a file is uploaded
        full_text = file.read().decode("utf-8")  # Read the contents of the file
    else:
        return "No text input or file provided"

    # Retrieve the number of questions from the form data
    num_questions = request.form.get('num_questions')

    # Check if num_questions is provided and is a valid integer
    if num_questions and num_questions.isdigit():
        num_questions = int(num_questions)
    else:
        # Set a default value for num_questions if not provided or invalid
        num_questions = 5

    summarized_text = summarize(full_text)
    keywords = get_nouns_multipartite(summarized_text)
    sentences = tokenize_sentences(summarized_text)
    keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
    key_distractor_list = {}

    # Define the maximum length for questions
    max_question_length = 200

    for keyword in keyword_sentence_mapping:
        if keyword_sentence_mapping[keyword]:
            wordsense = get_wordsense(keyword_sentence_mapping[keyword][0], keyword)
            if wordsense:
                distractors = get_distractors_wordnet(wordsense, keyword)
                if len(distractors) == 0:
                    distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
            else:
                distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors

    mcqs = []
    index = 1
    for each in key_distractor_list:
        sentence = keyword_sentence_mapping[each][0]
        # Truncate the sentence if it exceeds the maximum length
        sentence = sentence[:max_question_length]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub(" _______ ", sentence)
        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        random.shuffle(top4choices)
        correct_answer = each.capitalize()
        mcq = {
            'test_name': test_name,  # Store the test name
            'question': output,
            'choice1': top4choices[0],
            'choice2': top4choices[1],
            'choice3': top4choices[2],
            'choice4': top4choices[3],
            'correct_answer': correct_answer
        }
        mcqs.append(mcq)
        index += 1
        if index > num_questions:
            break

    # Store the generated MCQs into the database along with the test name
    for mcq in mcqs:
        new_mcq = MCQ(
            test_name=mcq['test_name'],  # Store the test name
            question=mcq['question'],
            choice1=mcq['choice1'],
            choice2=mcq['choice2'],
            choice3=mcq['choice3'],
            choice4=mcq['choice4'],
            correct_answer=mcq['correct_answer']
        )
        db.session.add(new_mcq)

    # Commit the session to the database
    db.session.commit()

    return redirect(url_for('mcqs_page'))



#main page and index page

#default master page route
@app.route('/')
def main_page():
    return render_template('main_page.html')


# Route for the index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    else:
        return render_template('index.html')

#Login and Sign up Module
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin_username = request.form['admin_username']
        admin_password = request.form['admin_password']
        
        # Check if the entered credentials match the hardcoded values
        if admin_username == 'tejas' and admin_password == 'tejas':
            session['admin_username'] = admin_username  # Store admin username in the session
            return redirect(url_for('admin_dashboard'))  # Redirect to admin dashboard
        else:
            return render_template('admin_login.html', message='Invalid admin username or password.')
    else:
        return render_template('admin_login.html')

# Routes for login and signup

# Route for user login
# Login route

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = Users.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = username
            # Redirect to select_exam route after successful login
            return redirect(url_for('select_exam'))  
        else:
            return render_template('login.html', message='Invalid username or password.')
    else:
        return render_template('login.html')



#signup for user
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Retrieve form data
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']  # Retrieve email field
    
        # Check if the username already exists
        existing_user = Users.query.filter_by(username=username).first()
        
        if existing_user:
            return render_template('signup.html', message='Username already exists. Please choose a different one.')
        else:
            # Insert the new user into the database
            new_user = Users(username=username, password=password, email=email)  # Include email field
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    else:
        return render_template('signup.html')

#admin_login model static 
@app.route('/view_result')
def view_result():
    # Check if the admin is logged in


    # Retrieve all exam scores from the database
    all_scores = exam_scores.query.all()

    # Render the admin dashboard template with the scores
    return render_template('test_result.html', all_scores=all_scores)

#admin dashboard
@app.route('/admin_dashboard')
def admin_dashboard():
    # Check if the admin is logged in
    if 'admin_username' not in session:
        return redirect(url_for('admin_login'))

    # Render the admin dashboard template
    return render_template('admin_dashboard.html')

#select test 
@app.route('/take_test')
def take_test():
    # Redirect to index.html
    return redirect(url_for('index'))



def admin_dashboard():
    # Render admin_dashboard.html
    return render_template('admin_dashboard.html')

#admin mcq fetching 
@app.route('/mcqs_page')
def mcqs_page():
    # Fetch all MCQs from the database
    mcqs = MCQ.query.all()
    # Render the MCQs page template and pass the fetched MCQs
    return render_template('mcqs.html', mcqs=mcqs)

#Exam module
@app.route('/select_exam')
def select_exam():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect(url_for('login'))

    # Render the user_exam.html file
    return render_template('select_exam.html')
    
@app.route('/get_test_names')
def get_test_names():
    test_names = MCQ.query.with_entities(MCQ.test_name).distinct().all()
    test_names = [name[0] for name in test_names]
    return jsonify(test_names)

@app.route('/user_exam.html')
def user_exam():
    # Retrieve the selected test name from the query parameter
    selected_test = request.args.get('test')
    
    # Check if the selected_test parameter is provided
    if selected_test:
        # Fetch MCQs for the selected test from the database
        mcqs = MCQ.query.filter_by(test_name=selected_test).all()
        
        # Render the user_exam.html template with the fetched MCQs
        return render_template('user_exam.html', mcqs=mcqs)
    else:
        # If no test is selected, you may want to handle this case accordingly
        return "No test selected"

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    if 'username' in session:
        username = session['username']
        score = 0
        total_questions_attempted = 0
        total_questions = int(request.form.get('total_rendered_questions', 0))

        # Retrieve submitted answers from the form
        for key, value in request.form.items():
            if key.startswith('answer_'):
                total_questions_attempted += 1
                question_id = int(key.split('_')[1])
                selected_option = value

                # Retrieve the correct answer for the question from the database
                mcq = MCQ.query.get(question_id)
                if mcq and selected_option == mcq.correct_answer:
                    score += 1

        # Calculate the score percentage based on the number of correct answers out of the total questions available
# Calculate the score percentage based on the number of correct answers out of the total questions available
        if total_questions > 0:
            score_percentage = (score / total_questions) * 100  # Calculate score percentage based on total questions
        else:
            score_percentage = 0


        # Store the user's score in the database
        exam_score = exam_scores(username=username, score=score_percentage)
        db.session.add(exam_score)
        db.session.commit()

        # Render the exam completed template with the score and total questions attempted
        return render_template('exam_completed.html', score=score_percentage, total_questions=total_questions, username=username)
    else:
        return redirect(url_for('login'))
    
with app.app_context():
    db.create_all()
if __name__ == '__main__':
    # Create all database tables before running the app


    app.run(debug=True, port=8000)
