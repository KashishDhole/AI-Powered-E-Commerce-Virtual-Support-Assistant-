import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import pyttsx3
import speech_recognition as sr
from threading import Thread
import time
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize the LLM with Google Generative AI
llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Initialize text-to-speech engine
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
    return engine

# Initialize speech recognition
@st.cache_resource
def init_stt():
    return sr.Recognizer()

def speak_text(text):
    """Convert text to speech"""
    try:
        engine = init_tts()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Speech error: {e}")

def listen_to_speech():
    """Convert speech to text"""
    recognizer = init_stt()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
        
        with microphone as source:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "Speech recognition service error"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.WaitTimeoutError:
        return "Listening timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment label
    if polarity > 0.1:
        sentiment = "Positive 😊"
        color = "#28a745"
    elif polarity < -0.1:
        sentiment = "Negative 😟"
        color = "#dc3545"
    else:
        sentiment = "Neutral 😐"
        color = "#6c757d"
    
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "color": color,
        "timestamp": datetime.now()
    }

def get_sentiment_summary(chat_history):
    """Get overall sentiment summary from chat history"""
    if not chat_history:
        return None
    
    user_messages = [chat["user"] for chat in chat_history]
    sentiments = [analyze_sentiment(msg) for msg in user_messages]
    
    avg_polarity = sum([s["polarity"] for s in sentiments]) / len(sentiments)
    avg_subjectivity = sum([s["subjectivity"] for s in sentiments]) / len(sentiments)
    
    positive_count = len([s for s in sentiments if s["polarity"] > 0.1])
    negative_count = len([s for s in sentiments if s["polarity"] < -0.1])
    neutral_count = len([s for s in sentiments if -0.1 <= s["polarity"] <= 0.1])
    
    return {
        "avg_polarity": avg_polarity,
        "avg_subjectivity": avg_subjectivity,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_messages": len(sentiments),
        "sentiments": sentiments
    }

def display_sentiment_analysis():
    """Display sentiment analysis dashboard"""
    st.subheader("📊 Sentiment Analysis Dashboard")
    
    if not st.session_state.get("chat_history"):
        st.info("No conversation data available for sentiment analysis.")
        return
    
    summary = get_sentiment_summary(st.session_state["chat_history"])
    
    if not summary:
        st.info("No sentiment data to display.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", summary["total_messages"])
    
    with col2:
        polarity_label = "Positive" if summary["avg_polarity"] > 0 else "Negative" if summary["avg_polarity"] < 0 else "Neutral"
        st.metric("Overall Sentiment", polarity_label, f"{summary['avg_polarity']:.2f}")
    
    with col3:
        st.metric("Subjectivity", f"{summary['avg_subjectivity']:.2f}", "Higher = More Opinion")
    
    with col4:
        satisfaction_score = max(0, min(100, (summary["avg_polarity"] + 1) * 50))
        st.metric("Satisfaction Score", f"{satisfaction_score:.0f}%")
    
    # Sentiment distribution pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Positive 😊', 'Neutral 😐', 'Negative 😟'],
            values=[summary["positive_count"], summary["neutral_count"], summary["negative_count"]],
            marker_colors=['#28a745', '#6c757d', '#dc3545']
        )])
        fig_pie.update_layout(title="Sentiment Distribution", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sentiment timeline
        df_sentiment = pd.DataFrame([
            {
                "Message": i+1,
                "Polarity": s["polarity"],
                "Sentiment": s["sentiment"]
            }
            for i, s in enumerate(summary["sentiments"])
        ])
        
        fig_timeline = px.line(df_sentiment, x="Message", y="Polarity", 
                              title="Sentiment Timeline",
                              color_discrete_sequence=['#007bff'])
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_timeline.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.5)
        fig_timeline.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.5)
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Message-wise sentiment analysis
    if st.checkbox("Show Message-wise Analysis"):
        st.subheader("Message-wise Sentiment Breakdown")
        
        for i, (chat, sentiment) in enumerate(zip(st.session_state["chat_history"], summary["sentiments"])):
            with st.expander(f"Message {i+1}: {sentiment['sentiment']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**User Message:**", chat["user"])
                    st.write("**Agent Response:**", chat["agent"][:100] + "..." if len(chat["agent"]) > 100 else chat["agent"])
                
                with col2:
                    st.write("**Sentiment:**", sentiment["sentiment"])
                    st.write("**Polarity:**", f"{sentiment['polarity']:.3f}")
                    st.write("**Subjectivity:**", f"{sentiment['subjectivity']:.3f}")
                    
                    # Color indicator
                    st.markdown(f"""
                    <div style='background-color: {sentiment['color']}; padding: 5px; border-radius: 3px; color: white; text-align: center; margin-top: 5px;'>
                        {sentiment['sentiment']}
                    </div>
                    """, unsafe_allow_html=True)

# Define the prompt template for the chatbot
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input", "product_problem"],
    template=(
        "You are a highly skilled customer care representative dedicated to resolving users' product-related issues. "
        "The user has described the following problem with their product: {product_problem}. "
        "Provide clear and practical solutions to address their concerns, including troubleshooting steps, potential timelines for resolution, warranty information, and any additional support they might need. "
        "Feel free to invent plausible details where necessary to offer a seamless customer service experience. "
        "Engage with the user in a polite, professional, and conversational tone, ensuring their satisfaction. "
        "Ensure that you give small answers in short points, that is not too long to read "
        "\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User: {user_input}\n\n"
        "Agent:"
    )
)

# Create the chain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Streamlit UI
st.set_page_config(page_title="Customer Care Chatbot", layout="centered")
st.title("🎧 Customer Care Chatbot")
st.write(
    """
    **Disclaimer**  
    We appreciate your engagement! Please note, this bot is designed to assist with product-related issues.  
    Type your queries below to get started.
    """
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "product_problem" not in st.session_state:
    st.session_state["product_problem"] = ""
if "show_sentiment" not in st.session_state:
    st.session_state["show_sentiment"] = False

# Navigation tabs
tab1, tab2 = st.tabs(["💬 Chat Support", "📊 Sentiment Analysis"])

with tab2:
    display_sentiment_analysis()

with tab1:
    # Step 1: Input the product and problem
    if not st.session_state["product_problem"]:
        st.subheader("Step 1: Describe Your Problem")
        product_problem_input = st.text_area("What issue are you facing with your product?", height=100)
        if st.button("Submit Problem", type="primary"):
            if product_problem_input.strip():
                st.session_state["product_problem"] = product_problem_input.strip()
                st.success(f"Problem noted: {st.session_state['product_problem']}")
                st.rerun()
            else:
                st.warning("Please describe your problem.")
    else:
        # Display current problem with sentiment
        problem_sentiment = analyze_sentiment(st.session_state["product_problem"])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**Current Issue:** {st.session_state['product_problem']}")
        with col2:
            st.markdown(f"""
            <div style='background-color: {problem_sentiment['color']}; padding: 8px; border-radius: 5px; color: white; text-align: center;'>
                Initial Sentiment: {problem_sentiment['sentiment']}
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("💬 Chat with Support")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state["chat_history"]):
                # User message with sentiment indicator
                user_sentiment = analyze_sentiment(chat["user"])
                
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(chat["user"])
                    with col2:
                        st.markdown(f"""
                        <small style='color: {user_sentiment["color"]}; font-weight: bold;'>
                            {user_sentiment["sentiment"]}
                        </small>
                        """, unsafe_allow_html=True)
                
                # Agent response
                with st.chat_message("assistant"):
                    st.write(chat["agent"])

        # Chat input
        if user_input := st.chat_input("Type your message here..."):
            if user_input.lower() == "quit":
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    st.write("Goodbye! Thank you for reaching out!")
                
                st.session_state["chat_history"].append({"user": user_input, "agent": "Goodbye! Thank you for reaching out!"})
                memory.clear()
            else:
                # Add user message to display with sentiment
                user_sentiment = analyze_sentiment(user_input)
                
                with st.chat_message("user"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(user_input)
                    with col2:
                        st.markdown(f"""
                        <small style='color: {user_sentiment["color"]}; font-weight: bold;'>
                            {user_sentiment["sentiment"]}
                        </small>
                        """, unsafe_allow_html=True)
                
                # Generate response from the LLM
                try:
                    chat_history_text = "\n".join(
                        [f"User: {item['user']}\nAgent: {item['agent']}" for item in st.session_state["chat_history"]]
                    )
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chain.run(
                                chat_history=chat_history_text,
                                user_input=user_input,
                                product_problem=st.session_state["product_problem"]
                            )
                        st.write(response.strip())
                        
                        # Speak the response if voice output is enabled
                        if st.session_state.get("enable_voice_output", False):
                            Thread(target=speak_text, args=(response.strip(),)).start()
                    
                    # Add to chat history
                    st.session_state["chat_history"].append({"user": user_input, "agent": response.strip()})
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Sidebar options
    with st.sidebar:
        st.subheader("Options")
        
        # Voice settings
        st.subheader("🎤 Voice Settings")
        enable_voice_output = st.checkbox("Enable Voice Response", value=True, key="enable_voice_output")
        enable_voice_input = st.checkbox("Enable Voice Input", value=False)
        
        if enable_voice_input:
            if st.button("🎙️ Start Voice Input"):
                with st.spinner("Listening... Please speak now"):
                    voice_text = listen_to_speech()
                if voice_text and not voice_text.startswith("Error") and not voice_text.startswith("Could not") and not voice_text.startswith("Listening timeout"):
                    st.session_state["voice_input"] = voice_text
                    st.success(f"Voice captured: {voice_text}")
                    st.rerun()
                else:
                    st.warning(f"Voice input failed: {voice_text}")
        
        # Sentiment quick view
        st.subheader("📊 Quick Sentiment View")
        if st.session_state.get("chat_history"):
            summary = get_sentiment_summary(st.session_state["chat_history"])
            if summary:
                st.metric("Messages", summary["total_messages"])
                satisfaction = max(0, min(100, (summary["avg_polarity"] + 1) * 50))
                st.metric("Satisfaction", f"{satisfaction:.0f}%")
                
                # Mini chart
                sentiment_counts = [summary["positive_count"], summary["neutral_count"], summary["negative_count"]]
                colors = ['#28a745', '#6c757d', '#dc3545']
                fig_mini = go.Figure(data=[go.Bar(
                    x=['😊', '😐', '😟'],
                    y=sentiment_counts,
                    marker_color=colors
                )])
                fig_mini.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_mini, use_container_width=True)
        else:
            st.info("No data yet")
        
        if st.button("🔄 Reset Chat", type="secondary"):
            st.session_state["product_problem"] = ""
            st.session_state["chat_history"] = []
            st.session_state.pop("voice_input", None)
            memory.clear()
            st.success("Chat reset successfully!")
            st.rerun()

    # Handle voice input if available (outside the else block)
    if st.session_state.get("product_problem") and "voice_input" in st.session_state and st.session_state["voice_input"]:
        voice_input = st.session_state["voice_input"]
        st.session_state.pop("voice_input", None)  # Clear voice input
        
        # Process voice input same as text input
        voice_sentiment = analyze_sentiment(voice_input)
        
        with st.chat_message("user"):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"🎤 {voice_input}")
            with col2:
                st.markdown(f"""
                <small style='color: {voice_sentiment["color"]}; font-weight: bold;'>
                    {voice_sentiment["sentiment"]}
                </small>
                """, unsafe_allow_html=True)
        
        if voice_input.lower() == "quit":
            with st.chat_message("assistant"):
                response_text = "Goodbye! Thank you for reaching out!"
                st.write(response_text)
                if st.session_state.get("enable_voice_output", False):
                    speak_text(response_text)
            
            st.session_state["chat_history"].append({"user": voice_input, "agent": response_text})
        else:
            try:
                chat_history_text = "\n".join(
                    [f"User: {item['user']}\nAgent: {item['agent']}" for item in st.session_state["chat_history"]]
                )
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chain.run(
                            chat_history=chat_history_text,
                            user_input=voice_input,
                            product_problem=st.session_state["product_problem"]
                        )
                    st.write(response.strip())
                    
                    # Speak the response if voice output is enabled
                    if st.session_state.get("enable_voice_output", False):
                        Thread(target=speak_text, args=(response.strip(),)).start()
                
                st.session_state["chat_history"].append({"user": voice_input, "agent": response.strip()})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
