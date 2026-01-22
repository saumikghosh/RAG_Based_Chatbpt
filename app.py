import os
import google.generativeai as genai
from pdfextracter import text_extracter
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Let's create the main page
st.title(':orange[CHATBOT:] :blue[AI assisted chatbot using RAG]')
tips = '''
Follow the steps to use this app
Step 1: Upload your PDF document in sidebar.
Step 2: Write a query and start the chat.'''
st.text(tips)
# Let's configure the model
# LLM
gemini_key = os.getenv("GOOGLE-API-19jan-key2")
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')
# Configure embedding model
embedding_model = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
# Let's create the sidebar
st.sidebar.title(':green[UPLOAD THE FILE]')
st.sidebar.subheader('Upload PDF file only')
pdf_file = st.sidebar.file_uploader('Upload here', type = ['pdf'])
if pdf_file:
    st.sidebar.success('File uploaded successfully')
    file_text = text_extracter(pdf_file)
    # Step 1 - Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(file_text)
    # Step 2 - Create vectorstore
    vector_store = FAISS.from_texts(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})

    # Step 3 - Retrieval
    def generate_content(query):
        retrieved_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrieved_docs])
        # step 4 augmenting
        augmented_prompt = f'''
        <Role> You are a helpful assistant using RAG.
        <Goal> Answer the query asked by the user. Here is the question: {query}
        <Context> Here are the documents retrieved from the vector database to support the answer which you have to generate {context}.
        '''
        # step 5 - generate
        response = model.generate_content(augmented_prompt)
        return response.text
    # Create chatbot in order to start the conversation.
    # To initialise a chat, we create history if not created
    if 'history' not in st.session_state:
        st.session_state.history = []
    # display the history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.info(f'User: {msg['text']}')
        else:
            st.warning(f'[CHATBOT:] {msg['text']}')
    # Take input from the user using streamlit form
    with st.form('Chatbot form', clear_on_submit=True):
        user_query = st.text_area('Ask Anything:')
        send = st.form_submit_button('Send')
    # Start the conversation & append output and query in history
    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'chatbot','text':generate_content(user_query)})
        st.rerun()


    


