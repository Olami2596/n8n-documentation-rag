import streamlit as st
import logging
import textwrap

from rag.retriever import DocumentRetriever
from rag.generator import AnswerGenerator

# Application logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="n8n Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. DESIGN SYSTEM

st.markdown(
    """
<style>

/* COLOUR PALETTE */

:root {
    --ivory: #F5F2E9;
    --ivory-dark: #EFEADE;
    --sage: #E2E9DC;
    --sage-dark: #D4DFCD;
    --forest: #214A38;
    --forest-light: #315D47;
    --sand: #E8DFCF;
    --ink: #26352C;
    --muted: #7D877B;
    --white: #FFFDF8;
    --border: #E6E0D4;
    --warning: #A66C24;
    --warning-bg: #F8EEDB;
}


/* GLOBAL LAYOUT */

.stApp {
    background: var(--ivory);
    color: var(--ink);
}

.block-container {
    max-width: 100%;
    padding: 1.5rem 2.5rem 3rem 2.5rem;
}

/* IMPORTANT: Do not hide the Streamlit toolbar.
   It contains the sidebar collapse/reopen controls. */

[data-testid="stToolbar"] {
    visibility: visible !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}


/* TYPOGRAPHY */

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    color: var(--ink) !important;
    letter-spacing: -0.7px;
}

h1 {
    font-size: 2.25rem !important;
    font-weight: 750 !important;
}

h2 {
    font-weight: 700 !important;
}

p {
    line-height: 1.7;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: var(--sage);
    border-right: 1px solid #D5DED0;
}

/* Allow the main workspace to expand when sidebar collapses */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0 !important;
    width: 0 !important;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 0 !important;
}

/* Reduce excessive top spacing */
[data-testid="stSidebar"] > div:first-child {
    padding: 0.4rem 1rem 1rem 1rem !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.3rem !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.2rem !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 3px 5px 17px 5px;
}

.brand-symbol {
    width: 40px;
    height: 40px;
    background: var(--forest);
    color: white;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    font-weight: 800;
}

.brand-name {
    color: var(--forest);
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1.1px;
    font-weight: 700;
}

.sidebar-label {
    color: #687868;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 15px 0 8px 5px;
}

.sidebar-status {
    background: rgba(255,255,255,0.55);
    border: 1px solid #D5DED0;
    border-radius: 10px;
    padding: 12px;
    margin-top: 8px;
}

.sidebar-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    margin: 8px 0;
    color: var(--ink);
}

.sidebar-online {
    color: #477C51;
    font-weight: 800;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    background: transparent;
    color: var(--ink);
    border: 1px solid transparent;
    border-radius: 8px;
    text-align: left;
    padding: 9px 12px;
    font-size: 13px;
    font-weight: 550;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.65);
    border-color: #C7D5C0;
    color: var(--forest);
}

[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--forest);
    color: white;
    border-color: var(--forest);
}

.sidebar-footer {
    border-top: 1px solid #D2DCCB;
    padding-top: 15px;
    margin-top: 25px;
    color: var(--muted);
    font-size: 10px;
    line-height: 1.8;
}


/* BRAND LABELS */

.eyebrow {
    color: var(--forest);
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.muted {
    color: var(--muted);
    font-size: 13px;
}


/* MAIN HEADER */

.workspace-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}

.header-status {
    background: #E4EDE0;
    color: var(--forest);
    border: 1px solid #D2DFCC;
    border-radius: 20px;
    padding: 7px 12px;
    font-size: 10px;
    font-weight: 750;
    white-space: nowrap;
}


/* WELCOME PANEL */

.welcome-panel {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 30px;
    margin: 18px 0 25px 0;
    box-shadow: 0 3px 14px rgba(43, 54, 40, 0.025);
}

.welcome-kicker {
    color: var(--forest);
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 2px;
    margin-bottom: 13px;
}

.welcome-title {
    color: var(--ink);
    font-size: 26px;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin-bottom: 10px;
}

.welcome-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
    max-width: 620px;
    margin-bottom: 0;
}


/* BUTTONS */

.stButton button {
    border: 1px solid var(--border);
    background: var(--white);
    color: var(--forest);
    border-radius: 9px;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton button:hover {
    background: var(--sage);
    border-color: #B8C9B2;
    color: var(--forest);
}

.stButton button[kind="primary"] {
    background: var(--forest);
    color: white;
    border: 1px solid var(--forest);
}

.stButton button[kind="primary"]:hover {
    background: var(--forest-light);
    border-color: var(--forest-light);
    color: white;
}


/* CHAT MESSAGES */

[data-testid="stChatMessage"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 17px;
    margin-bottom: 13px;
}


/* CHAT INPUT */

[data-testid="stChatInput"] {
    background: var(--white);
    border: 1px solid #D8DFD1;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(33,74,56,0.04);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--forest);
}


/* RESEARCH PANEL */

.research-panel {
    background: #F0EDE4;
    border: 1px solid #E3DDCF;
    border-radius: 14px;
    padding: 19px;
    margin-top: 15px;
}

.panel-heading {
    color: var(--ink);
    font-size: 17px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.panel-description {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.6;
    margin-top: 5px;
}


/* SOURCE CARDS */

.source-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 13px;
    margin: 10px 0;
}

.source-title {
    color: var(--ink);
    font-size: 12px;
    font-weight: 750;
    line-height: 1.5;
}

.source-meta {
    color: var(--muted);
    font-size: 10px;
    margin-top: 7px;
}


/* STATUS BADGES */

.warning-panel {
    background: var(--warning-bg);
    border: 1px solid #EAD9B8;
    border-radius: 10px;
    padding: 13px;
    color: #79551F;
    font-size: 11px;
    line-height: 1.7;
}


/* EXPANDERS */

[data-testid="stExpander"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
}


/* METRICS */

[data-testid="stMetric"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 13px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--forest);
}


/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 10px;
}


/* DIVIDERS */

hr {
    border-color: var(--border);
}


/* RESPONSIVE LAYOUT */

@media (max-width: 1000px) {

    .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# 3. SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# 4. LOAD RAG COMPONENTS


@st.cache_resource(show_spinner=False)
def load_rag_components():

    retriever = DocumentRetriever(top_k=5, namespace="n8n-full-corpus")
    generator = AnswerGenerator()

    return retriever, generator


# 5. HELPER FUNCTIONS


def render_sources(sources):

    if not sources:
        st.caption("No supporting documents were returned.")
        return

    for source in sources:

        with st.container(border=True):

            st.markdown(f"**[{source['reference']}] {source['title']}**")

            status = source.get("status", "unspecified")

            if status == "deprecated":
                st.error("Deprecated documentation")
            else:
                st.caption(f"Status: {status}")

            st.markdown(f"[Open original documentation]({source['url']})")


def render_assistant_message(message):

    st.markdown(message["content"])

    sources = message.get("sources", [])

    if sources:

        with st.expander(f"VIEW SUPPORTING DOCUMENTATION | {len(sources)} SOURCES"):
            render_sources(sources)

    if message.get("has_deprecated_sources"):

        st.warning(
            "Deprecated documentation was detected during retrieval. "
            "It was excluded from answer generation."
        )

        for source in message.get("deprecated_sources", []):
            st.caption(f"Deprecated source: {source['title']}")


def ask_question(question):

    st.session_state.pending_question = question
    st.rerun()


# 6. SIDEBAR

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-symbol">N</div>
            <div>
                <div class="brand-name">n8n Intelligence</div>
                <div class="brand-subtitle">DOCUMENTATION SYSTEM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)

    if st.button("+  New conversation", use_container_width=True, type="primary"):

        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="sidebar-label">EXPLORE DOCUMENTATION</div>', unsafe_allow_html=True
    )

    topics = {
        "Workflows": "How do I create and execute an n8n workflow?",
        "Integrations": "How do integrations work in n8n?",
        "Deployment": "What are the different ways to deploy n8n?",
        "n8n API": "How can I interact with n8n using its API?",
    }

    for topic, question in topics.items():

        if st.button(f">  {topic}", key=f"nav_{topic}", use_container_width=True):
            ask_question(question)

    st.markdown(
        '<div class="sidebar-label">SYSTEM STATUS</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-status">
            <div class="sidebar-status-row">
                <span>RAG Engine</span>
                <span class="sidebar-online">Ready</span>
            </div>
            <div class="sidebar-status-row">
                <span>Vector Database</span>
                <span class="sidebar-online">Connected</span>
            </div>
            <div class="sidebar-status-row">
                <span>Embeddings</span>
                <span>MiniLM | 384D</span>
            </div>
            <div class="sidebar-status-row">
                <span>Language Model</span>
                <span>GPT-OSS 20B</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">CONVERSATION</div>', unsafe_allow_html=True)

    completed = sum(
        1 for message in st.session_state.messages if message["role"] == "user"
    )

    st.caption(f"{completed} question(s) in this session")

    st.markdown(
        """
        <div class="sidebar-footer">
            INDEPENDENT PORTFOLIO PROJECT<br>
            Built with Python, Pinecone and Groq.<br>
            Not affiliated with n8n.
        </div>
        """,
        unsafe_allow_html=True,
    )


# 7. MAIN HEADER

st.markdown(
    """
    <div class="workspace-header">
        <div>
            <div class="eyebrow">N8N / KNOWLEDGE EXPLORER</div>
        </div>
        <div class="header-status">RAG SYSTEM READY</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Your documentation workspace.")

st.markdown(
    """
    <p class="muted">
    Explore n8n concepts, understand technical processes,
    and find answers backed by documentation.
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()


# 8. MAIN WORKSPACE COLUMNS

main_column, research_column = st.columns([1.9, 1], gap="large")


# 9. CENTRE WORKSPACE

with main_column:

    if not st.session_state.messages:

        st.html(
            textwrap.dedent("""
            <div class="welcome-panel">
                <div class="welcome-kicker">
                    YOUR KNOWLEDGE WORKSPACE
                </div>

                <div class="welcome-title">
                    What would you like to understand?
                </div>

                <p class="welcome-description">
                    Ask technical questions about n8n workflows,
                    integrations, deployment and APIs.

                    Explore the documentation and get answers
                    supported by original source material.
                </p>
            </div>
            """)
        )

        st.markdown("### Explore a starting point")

        st.markdown(
            '<p class="muted">Choose a topic to begin exploring.</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("01  |  Workflows", use_container_width=True):
                ask_question("How do I create and execute an n8n workflow?")

            if st.button("02  |  Integrations", use_container_width=True):
                ask_question("How do integrations work in n8n?")

        with col2:

            if st.button("03  |  Deployment", use_container_width=True):
                ask_question("What are the different ways to deploy n8n?")

            if st.button("04  |  API", use_container_width=True):
                ask_question("How can I interact with n8n using its API?")

    # Display conversation history

    for message in st.session_state.messages:

        avatar = "👤" if message["role"] == "user" else "🌿"

        with st.chat_message(message["role"], avatar=avatar):

            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])


# 10. CHAT INPUT

with main_column:

    typed_question = st.chat_input("Ask about workflows, integrations, deployment...")

question = st.session_state.pending_question or typed_question

st.session_state.pending_question = None

# 11. PROCESS USER QUESTION

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with main_column:

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🌿"):

            try:

                with st.spinner("Searching documentation and preparing your answer..."):

                    # Load the actual RAG component instances
                    document_retriever, answer_generator = load_rag_components()

                    # Collect previous conversation messages.
                    # Exclude the latest message because it is the current question.
                    conversation_history = [
                        {"role": message["role"], "content": message["content"]}
                        for message in st.session_state.messages[:-1]
                        if message.get("role") in ("user", "assistant")
                        and message.get("content")
                    ]

                    # Rewrite the question using conversation history
                    rewritten_question = answer_generator.rewrite_question(
                        question, conversation_history=conversation_history
                    )

                    # Retrieve documentation using the rewritten question
                    retrieval_response = document_retriever.search(rewritten_question)

                    # Generate an answer using documentation and conversation history
                    response = answer_generator.generate_answer(
                        question,
                        retrieval_response,
                        conversation_history=conversation_history,
                    )

                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "has_deprecated_sources": response["has_deprecated_sources"],
                    "deprecated_sources": response["deprecated_sources"],
                }

                render_assistant_message(assistant_message)

                st.session_state.messages.append(assistant_message)

            except Exception as error:

                # Record technical details in application logs.
                logger.exception("Failed to process user question.")

                # Provide a friendly message to the user.
                error_message = (
                    "I couldn't process your question right now. "
                    "Please try again in a little while."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )


# 12. RIGHT RESEARCH PANEL

with research_column:

    st.html(
        textwrap.dedent("""
        <div class="research-panel">

            <div class="eyebrow">RESEARCH PANEL</div>

            <div class="panel-heading">
                Knowledge sources
            </div>

            <div class="panel-description">
                Documentation supporting your latest answer.
            </div>

        </div>
        """)
    )

    latest_assistant = next(
        (
            message
            for message in reversed(st.session_state.messages)
            if message["role"] == "assistant"
        ),
        None,
    )

    if latest_assistant:

        sources = latest_assistant.get("sources", [])

        st.metric("Retrieved source pages", len(sources))

        render_sources(sources)

        if latest_assistant.get("has_deprecated_sources"):

            st.markdown(
                """
                <div class="warning-panel">
                    <strong>Warning: Deprecated content detected</strong><br>
                    Some retrieved documentation was deprecated
                    and excluded from answer generation.
                </div>
                """,
                unsafe_allow_html=True,
            )

            for source in latest_assistant.get("deprecated_sources", []):
                st.caption(source["title"])

    else:

        st.info(
            "Your supporting documents will appear here "
            "after you ask your first question."
        )

        st.markdown("### How it works")

        st.markdown("""
            **01 | Retrieve**

            Find relevant documentation.

            **02 | Analyse**

            Provide supporting content to the language model.

            **03 | Answer**

            Generate an answer with source references.
            """)

    st.divider()

    st.markdown("### Documentation policy")

    st.caption(
        "Answers are grounded in retrieved documentation. "
        "Deprecated documents are excluded from answer generation."
    )
