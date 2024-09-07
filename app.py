import streamlit as st
import json, os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(page_title="DocumentGPT", page_icon="❓")
st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "string",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based on only the difficulty and the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    If the answer is correct, set the value to true; otherwise, set it to false.
    
    If the difficulty is set to HARD, create a difficult questions; if it is set to EASY, create an easy questions.

    Your turn!
    
    Difficulty: {difficulty}

    Context: {context}
""",
        )
    ]
)

with st.sidebar:
    openai_api_key = st.text_input("Input OpenAI API Key:")

if openai_api_key != "":
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=openai_api_key,
    ).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, difficulty, topic):
    chain = questions_prompt | llm
    return chain.invoke({"context": _docs, "difficulty": difficulty})

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=3)
    docs = retriever.get_relevant_documents(term)
    return docs

topic = None
with st.sidebar:
    docs = None

    difficulty = st.selectbox("Choose difficulty level", ("Easy", "Hard"))

    choice = st.selectbox("Choose what you want to use.", ("File", "Wikipedia Article"))
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, .pdf file", type=["docx", "txt", "pdf"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, difficulty, topic if topic else file.name)
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)
    success_cnt = 0

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": "true"} in question["answers"]:
                success_cnt += 1
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()

    if len(response["questions"]) == success_cnt:
        st.balloons()
