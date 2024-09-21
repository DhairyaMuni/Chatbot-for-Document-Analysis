import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from groq import Groq
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional
import voyageai
from langchain.embeddings.base import Embeddings
import numpy as np
from htmlTemplates import css, bot_template, user_template
from pydantic import Field
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load environment variables
load_dotenv()

# Configuration for each department with a shared API key
class DepartmentConfig:
    def __init__(self, name, model_name, embedding_model):
        self.name = name
        self.api_key = os.getenv("VOYAGE_API_KEY")  # Unified API Key
        self.model_name = model_name
        self.embedding_model = embedding_model

# Define configurations for each department
departments_config = {
    "General": DepartmentConfig(
        name="General",
        model_name="llama3-70b-8192",
        embedding_model="voyage-large-2-instruct"
    ),
    "HR": DepartmentConfig(
        name="HR",
        model_name="llama3-70b-8192-hr",
        embedding_model="voyage-large-2-instruct"
    ),
    "Finance": DepartmentConfig(
        name="Finance",
        model_name="llama3-70b-8192-finance",
        embedding_model="voyage-finance-2"
    ),
    "IT": DepartmentConfig(
        name="IT",
        model_name="llama3-70b-8192-it",
        embedding_model="voyage-large-2-instruct"
    ),
    "Marketing": DepartmentConfig(
        name="Marketing",
        model_name="llama3-70b-8192-marketing",
        embedding_model="voyage-large-2-instruct"
    ),
    "Operations": DepartmentConfig(
        name="Operations",
        model_name="llama3-70b-8192-operations",
        embedding_model="voyage-large-2-instruct"
    ),
    "Law": DepartmentConfig(
        name="Law",
        model_name="llama3-70b-8192-law",
        embedding_model="voyage-law-2"
    ),
    "Code": DepartmentConfig(
        name="Code",
        model_name="llama3-70b-8192-code",
        embedding_model="voyage-code-2"
    )
}

class VoyageAIEmbeddings(Embeddings):
    """Embeddings class to handle document and query embeddings with VoyageAI."""
    
    def __init__(self, voyage_client, model_name="voyage-large-2-instruct", batch_size=128):
        self.client = voyage_client
        self.model_name = model_name
        self.dimension = None
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                embeddings = self.client.embed(batch, model=self.model_name, input_type="document")
                if hasattr(embeddings, 'embeddings'):
                    numpy_embeddings = [np.array(emb) for emb in embeddings.embeddings]
                elif isinstance(embeddings, list):
                    numpy_embeddings = [np.array(emb) for emb in embeddings]
                else:
                    raise ValueError(f"Unexpected embeddings type: {type(embeddings)}")
                
                all_embeddings.extend(numpy_embeddings)
                print(f"Processed batch {i//self.batch_size + 1} of size {len(batch)}")
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")

        if self.dimension is None and all_embeddings:
            self.dimension = len(all_embeddings[0])
        
        return [emb.tolist() for emb in all_embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            embedding = self.client.embed([text], model=self.model_name, input_type="query")
            if hasattr(embedding, 'embeddings'):
                numpy_embedding = np.array(embedding.embeddings[0])
            elif isinstance(embedding, list):
                numpy_embedding = np.array(embedding[0])
            else:
                raise ValueError(f"Unexpected query embedding type: {type(embedding)}")
            
            if self.dimension is None:
                self.dimension = len(numpy_embedding)
            return numpy_embedding.tolist()
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            raise

    def embed_batch_documents(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Embed a batch of documents."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_batch_queries(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Embed a batch of queries."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [self.embed_query(text) for text in batch]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

class GroqLLM(LLM):
    """LLM class to handle conversational AI with Groq."""
    
    client: Any = Field(default_factory=lambda: Groq(api_key=os.getenv("GROQ_API_KEY2")))  # Unified API Key
    model_name: str = "llama3-70b-8192"
    department: str = "General"
    
    def __init__(self, **data):
        super().__init__(**data)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY2"))  # Unified API Key
    
    @property
    def system_prompt(self):
        """Return the system prompt for the specific department."""
        prompts = {
            "Code":"You are an AI agent with extensive knowledge in coding. Assist the user with contextual coding queries, providing comprehensive guidance and solutions.",
            "HR": "You are an HR assistant specializing in labor law and workplace policies. Provide comprehensive information on human resources, employee policies, and workplace regulations. Respond based on the content provided, and include additional relevant details as necessary.",
            "Finance": "You are a finance assistant specializing in financial reports, budgets, and accounting practices. Offer detailed information on financial analysis, reporting, and accounting principles. Respond based on the content provided, and include additional relevant insights as necessary.",
            "IT": "You are an IT assistant specializing in technology, software, and IT infrastructure. Provide detailed information on IT systems, software solutions, and infrastructure management. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Marketing": "You are a marketing assistant specializing in marketing strategies, campaigns, and market analysis. Offer comprehensive information on digital marketing, advertising tactics, and consumer behavior analysis. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Operations": "You are an operations assistant specializing in business processes, supply chain management, and operational efficiency. Provide detailed information on logistics, process optimization, and supply chain strategies. Respond based on the content provided, and include additional relevant insights as necessary.",
            "Law": "You are a legal assistant specializing in laws, regulations, contracts, and legal procedures. Your responses should not be considered as legal advice. Provide detailed information on legal frameworks, contract analysis, and procedural guidelines. Respond based on the content provided, and include additional relevant insights as necessary.",
            "General": "You are a helpful assistant dedicated to providing informative and supportive responses. Offer assistance across various topics, ensuring accuracy and clarity in your replies. Respond based on the context provided, and provide additional relevant information when beneficial."
        }
        return prompts.get(self.department, prompts["General"])
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call the LLM with the given prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters for the LLM."""
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}

def get_pdf_texts(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

def chunk_text(text, max_tokens=5000):
    """Split text into chunks of `max_tokens`."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_count = len(word)
        if current_token_count + token_count < max_tokens:
            current_chunk.append(word)
            current_token_count += token_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_token_count = token_count
    chunks.append(" ".join(current_chunk))
    return chunks


def get_vector_store(text_chunks, department):
    """Create a vector store using embeddings for the specified department."""
    vo = voyageai.Client(api_key=os.getenv("API_KEY"))  # Use the unified API key
    if isinstance(text_chunks, list):
        input_texts = text_chunks
    else:
        input_texts = [text_chunks]
    
    try:
        full_text = " ".join(input_texts)
        model_name = get_embedding_model(department, full_text)
        
        embeddings = VoyageAIEmbeddings(vo, model_name=model_name, batch_size=128)
        all_embeddings = embeddings.embed_batch_documents(input_texts)
        
        print(f"Using embedding model: {model_name}")
        print(f"Embedding dimension: {embeddings.dimension}")
        print(f"Total embeddings: {len(all_embeddings)}")
        
        text_embeddings = list(zip(input_texts, all_embeddings))
        
        vectorstore = FAISS.from_embeddings(text_embeddings, embeddings)
        vectorstore.embedding_function = embeddings
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        print(f"Detailed error: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def get_embedding_model(department, text_content):
    """Select the embedding model based on the department."""
    config = departments_config.get(department)
    return config.embedding_model

def get_conversation_chain(vectorstore, department):
    """Create a conversation chain for the specified department."""
    config = departments_config.get(department)
    llm = GroqLLM(department=department, model_name=config.model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    conversation_chain.retriever.vectorstore.embedding_function = vectorstore.embedding_function
    return conversation_chain

def handle_userinput(user_question, department):
    """Handle user input and provide responses based on the department."""
    if department not in st.session_state.conversation:
        st.warning(f"No conversation chain found for {department}. Please process documents first.")
        return

    response = st.session_state.conversation[department]({"question": user_question})
    st.session_state.chat_history[department] = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history[department]):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def process_documents(uploaded_files, department):
    """Process uploaded documents for a specific department."""
    if uploaded_files:
        with st.spinner(f"Processing {department} documents"):
            if any(file.type == 'text/csv' for file in uploaded_files):
                for file in uploaded_files:
                    if file.type == 'text/csv':
                        df = pd.read_csv(file)
                        st.write(f"Uploaded CSV Data for {department}:")
                        st.write(df)
                        st.session_state.dataframes[department] = df
                return None
            else:
                raw_text = get_pdf_texts(uploaded_files)
                if raw_text:
                    text_chunks = chunk_text(raw_text)
                    vectorstore = get_vector_store(text_chunks, department)
                    if vectorstore:
                        conversation_chain = get_conversation_chain(vectorstore, department)
                        st.session_state.conversation[department] = conversation_chain
                        return vectorstore
                    else:
                        st.error("Error creating vector store. Please try again.")
                else:
                    st.error("No text could be extracted from the uploaded PDFs.")
    else:
        st.error("Please upload at least one document.")
    return None

def extract_column_name(prompt):
    """Extract column name from the user's prompt."""
    keywords = ["of", "on", "from"]
    words = prompt.split()
    for i, word in enumerate(words):
        if word in keywords and i < len(words) - 1:
            return words[i + 1]
    return None

def analyze_csv(department, prompt):
    """Analyze CSV data based on user's prompt."""
    if department not in st.session_state.dataframes:
        st.warning(f"No CSV data found for {department}. Please upload a CSV file first.")
        return

    df = st.session_state.dataframes[department]
    prompt_lower = prompt.lower()
    column = extract_column_name(prompt)

    if column is not None and column in df.columns:
        if "mean" in prompt_lower or "average" in prompt_lower:
            result = f"Mean of '{column}': {df[column].mean()}"
        elif "median" in prompt_lower:
            result = f"Median of '{column}': {df[column].median()}"
        elif "mode" in prompt_lower:
            mode_values = df[column].mode()
            result = f"Mode of '{column}': {', '.join(map(str, mode_values))}" if not mode_values.empty else f"No mode found in '{column}'."
        elif "histogram" in prompt_lower:
            plt.figure(figsize=(10, 6))
            plt.hist(df[column], bins=20)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot(plt)
            result = f"Histogram of {column} has been generated."
        elif "scatterplot" in prompt_lower:
            x_column = st.selectbox("Select the X-axis column:", df.columns)
            plt.figure(figsize=(10, 6))
            plt.scatter(df[x_column], df[column])
            plt.title(f"Scatter plot of {x_column} vs {column}")
            plt.xlabel(x_column)
            plt.ylabel(column)
            st.pyplot(plt)
            result = f"Scatter plot of {x_column} vs {column} has been generated."
        elif "count" in prompt_lower:
            result = f"Count of '{column}': {df[column].count()}"
        elif "sum" in prompt_lower:
            result = f"Sum of '{column}': {df[column].sum()}"
        elif "null" in prompt_lower:
            result = f"Null value count in '{column}': {df[column].isnull().sum()}"
        elif "min" in prompt_lower:
            result = f"Min value in '{column}': {df[column].min()}"
        elif "max" in prompt_lower:
            result = f"Max value in '{column}': {df[column].max()}"
        elif "correlation" in prompt_lower:
            corr_matrix = df.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            st.pyplot(plt)
            result = "Correlation heatmap has been generated."
        else:
            result = "Unsupported analysis prompt. Please try a different query."
    else:
        result = f"Column '{column}' not found in the dataset. Available columns are: {', '.join(df.columns)}"

    st.write(result)


def main():
    """Main function to run the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="Departmental Document Chat", page_icon=":office:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    
    st.header("Departmental Document Chat :office:")
    
    departments = list(departments_config.keys())
    selected_department = st.selectbox("Select your department:", departments)
    
    user_question = st.text_input(f"Ask a question about {'your documents' if selected_department == 'General' else f'{selected_department} documents'}:")
    if user_question:
        if selected_department in st.session_state.dataframes:
            analyze_csv(selected_department, user_question)
        else:
            handle_userinput(user_question, selected_department)
    
    with st.sidebar:
        st.subheader(f"{'Your' if selected_department == 'General' else selected_department} Documents")
        uploaded_files = st.file_uploader(
            f"Upload {'documents' if selected_department == 'General' else f'{selected_department} documents'} here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf', 'csv']
        )
        if st.button("Process"):
            result = process_documents(uploaded_files, selected_department)
            if result is not None:
                st.success(f"Processing complete! You can now ask questions about your {selected_department} documents.")
            elif selected_department in st.session_state.dataframes:
                st.success(f"CSV data for {selected_department} loaded successfully. You can now analyze this data.")

if __name__ == '__main__':
    main()