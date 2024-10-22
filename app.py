import os
import streamlit as st
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

st.title("Chat With Your Resume ! 😊 ")
# st.write("Upload your resume and ask questions to analyze its strengths, weaknesses, and more.")

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm),
)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


if not st.session_state.chat_active:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        resume_data = extract_text_from_pdf(uploaded_file)
        
        template = PromptTemplate(
            template="You are a resume analysis assistant with expertise in extracting key details from user resumes. \
            Your goal is to parse the provided resume and return structured data, including sections such as: \
            - Personal Information (Name, Contact Details) \
            - Work Experience (Job Titles, Companies, Dates, Responsibilities) \
            - Education (Degrees, Institutions, Graduation Dates) \
            - Skills (Technical Skills, Soft Skills) \
            - Certifications and Awards \
            - Other relevant information. \
            Please extract all important information from this resume: {resume_data}",
            input_variables=['resume_data']
        )
        string_template = template.format(resume_data=resume_data)
        llm_parsed_resume = conversation.predict(input=string_template)
        
        st.session_state.resume_data = llm_parsed_resume
        st.session_state.chat_active = True

        st.session_state.messages.append({"role": "assistant", "content": "Resume Uploaded and Analyzed Successfully! You can now ask me questions about it."})
        st.rerun()
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    user_input_template = PromptTemplate(
        template="You are a resume analysis expert. Given the parsed resume {llm_parsed_resume}, help the user answer their questions. Here is the question: {user_input}",
        input_variables=['llm_parsed_resume', 'user_input']
    )

    if prompt := st.chat_input("Ask me about your resume :)"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        user_input_string_template = user_input_template.format(llm_parsed_resume=st.session_state.resume_data, user_input=prompt)
        response = conversation.predict(input=user_input_string_template)
        # TODO : write_steam(generator)
        with st.chat_message("assistant"):
          st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Reset"):
    st.session_state.messages = []
    st.session_state.resume_data = None
    st.session_state.chat_active = False
    st.rerun()