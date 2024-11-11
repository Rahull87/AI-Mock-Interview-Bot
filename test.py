import os
import tempfile
import streamlit as st
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summerizer import summarize_resume, summarize_jd
import tempfile
from difflib import SequenceMatcher 
load_dotenv()
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# Configure the API key for Google Gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyDJQi4UsvTyJtA4kNuL5e-b6M61WMVwL2M"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text):
    model = genai.GenerativeModel("gemini-1.5-flash",
                                  generation_config={"response_mime_type": "application/json"}
                                  )
    response = model.generate_content(input_text)
    return response.text

def generate_interview_question(interview_round, resume_text, job_description):
    input_data = {
        "Interview Round": interview_round,
        "Resume": resume_text,
        "Job Description": job_description,
    }
    
    prompt = f"""
    You are an AI assistant designed to help users prepare for job interviews by simulating different types of interview rounds. 
    Your task is to generate one interview question based on the provided resume, job description, and 
    the selected interview round. Along with the question, generate a sample answer and provide feedback on its strengths and weaknesses.

    **Avoid generating questions of the same type in consecutive responses.**

    Interview Round: {input_data['Interview Round']}
    Job Description: {input_data['Job Description']}
    Resume: {input_data['Resume']}

    Please return the question, sample answer, and feedback in the following format:
    {{
        "question": "str",
        "sample_answer": "str",
        "feedback": "str"
    }}
    """

    raw_response = get_gemini_response(prompt)
    
    #st.write("Raw Response:", raw_response)
    
    try:
        qa_pair = json.loads(raw_response)
        return qa_pair['question'], qa_pair['sample_answer']
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response: {e}")
        return None, None

def evaluate_answer(user_answer, model_answer):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_answer, model_answer])
    cosine_sim = cosine_similarity(vectors)[0][1]
    similarity_score = round(cosine_sim * 100, 2)
    return similarity_score

# Streamlit app setup
st.set_page_config(page_title="Mock Interview Bot", layout="wide")
st.title("Mock Interview Bot")

# Interview rounds selection
interview_rounds = [
    "Managerial Round",
    "Behavioral Round",
    "Technical Round"
]
interview_round = st.selectbox("Select Type of Interview", interview_rounds, key="interview_round")

# Upload files: Resume and Job Description (JD)
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf", key="resume_file")
job_description_file = st.file_uploader("Upload Your Job Description (TXT)", type="txt", key="job_description")

# Initialize session state for tracking user progress
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "questions_list" not in st.session_state:
    st.session_state.questions_list = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "summarized_job_description" not in st.session_state:
    st.session_state.summarized_job_description = ""
if "model_answer" not in st.session_state:
    st.session_state.model_answer = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Helper function to read and decode job description
def read_job_description(file):
    try:
        #content = file.read().decode("utf-8")
        content = file.read().decode("utf-8", errors="ignore")

        #st.write("Job Description Content:", content)
        return content
    except UnicodeDecodeError:
        content = file.read().decode("ISO-8859-1")
        #st.write("Job Description Content with ISO-8859-1:", content)
        return content

# Button to start the interview preparation
if st.button("Start Interview Preparation"):
    if interview_round and uploaded_file and job_description_file:
        with st.spinner("Generating questions..."):
            try:
                # Handle resume file (PDF)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_file_path = tmp_file.name

                # Summarize resume and JD
                st.session_state.resume_text = summarize_resume(temp_file_path)

                # Read the job description text
                job_description_text = read_job_description(job_description_file)
                
                if job_description_text:
                    st.session_state.summarized_job_description = summarize_jd(job_description_text)

                    #st.write("### Summarized Job Description")
                    #st.write(st.session_state.summarized_job_description)

                    # Generate interview question based on the inputs
                    question, sample_answer = generate_interview_question(
                        interview_round,
                        st.session_state.resume_text,
                        st.session_state.summarized_job_description
                    )

                    if question:
                        st.session_state.current_question = question
                        st.session_state.model_answer = sample_answer
                        st.session_state.question_count = 1

                else:
                    st.error("Failed to read the job description. Please check the file.")

                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both resume and job description files.")

# Display previous questions and answers
if st.session_state.questions_list:
    st.write("### Previous Questions and Answers")
    for qa in st.session_state.questions_list:
        st.write(f"**Question {qa['question_number']}:** {qa['question']}")
        st.write(f"**User Answer:** {qa['user_answer']}")
        st.write(f"**Model Answer:** {qa['model_answer']}")
        st.write(f"**Similarity Score:** {qa['similarity_score']}%")
        st.write("---")

# Display current question
if st.session_state.current_question:
    st.write(f"**Question {st.session_state.question_count}:** {st.session_state.current_question}")

    with st.form(key='answer_form', clear_on_submit=True):
        user_answer = st.text_area("Your Answer", key="user_answer")
        submit_button = st.form_submit_button("Submit Answer")

        if submit_button:
            if user_answer:
                score = evaluate_answer(user_answer, st.session_state.model_answer)

                st.session_state.questions_list.append({
                    "question_number": st.session_state.question_count,
                    "question": st.session_state.current_question,
                    "user_answer": user_answer,
                    "model_answer": st.session_state.model_answer,
                    "similarity_score": score
                })

                # Display the evaluation results
                st.write(f"**Evaluation Score:** {score}%")
                st.write(f"**Model's Suggested Answer:** {st.session_state.model_answer}")

                # Generate the next question
                with st.spinner("Generating next question..."):
                    try:
                        question, sample_answer = generate_interview_question(
                            interview_round,
                            st.session_state.resume_text,
                            st.session_state.summarized_job_description
                        )

                        if question:
                            st.session_state.current_question = question
                            st.session_state.model_answer = sample_answer
                            st.session_state.question_count += 1
                        else:
                            st.write("You have completed all the questions.")

                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please provide an answer to proceed.")
