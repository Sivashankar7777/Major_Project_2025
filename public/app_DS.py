import os
from apikey import apikey

# Streamlit for web app
import streamlit as st

# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning models and utilities
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    mean_squared_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Time series analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# LangChain imports for LLMs and utilities
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.utilities import WikipediaAPIWrapper

# dotenv for environment variables
from dotenv import load_dotenv, find_dotenv

from arima_module.arima_code import main_arima 



# OpenAI Key
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Streamlit App Title
st.title(':red[_Info_] :blue[_Mind_] 🤖')

# Sidebar
with st.sidebar:
    st.write(':red[_AI Assistant_] for :blue[_Data Science_] ')
    st.caption("""
        **Every data science journey starts with a dataset. 
        Upload your CSV or image file, and we'll explore it together.**
    """)
    st.divider()
    st.caption("<p style='text-align:center'> made by Siva, Pranav and Parvathi</p>", unsafe_allow_html=True)

# Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])


if st.session_state.clicked[1]:
    tab1, tab2 = st.tabs([":red[_Data Analysis_] and :blue[_Data Science_]", "ChatBox"])
    
    with tab1:
        user_file = st.file_uploader("Upload your CSV or Image file", type=["csv", "png", "jpg", "jpeg"])
        if user_file is not None:
            user_file.seek(0)
            df = pd.read_csv(user_file, low_memory=True)

            # LLM model
            llm = OpenAI(temperature=0)

            # Create pandas agent with dangerous code enabled
            try:
                pandas_agent = create_pandas_dataframe_agent(
                    llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True
                )
            except ValueError as ve:
                st.error(f"Error creating pandas dataframe agent: {ve}")
                st.stop()

            # Function to handle general EDA tasks
            @st.cache_data
            def function_agent():
                st.write("**Data Overview**")
                st.write("The first rows of your dataset look like this:")
                st.write(df.head())  # Manual inspection to avoid parsing issues

                st.write("**Data Cleaning**")
                try:
                    columns_df = pandas_agent.run("What are the meanings of the columns?")
                    st.write(columns_df)

                    missing_values = pandas_agent.run(
                        "How many missing values does this dataframe have? Start the answer with 'There are'."
                    )
                    st.write(missing_values)

                    duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                    st.write(duplicates)
                except ValueError as ve:
                    st.warning(f"Error during data inspection: {ve}")

                st.write("**Data Summarisation**")
                st.write(df.describe())

                try:
                    correlation_analysis = pandas_agent.run(
                        "Calculate correlations between numerical variables to identify potential relationships."
                    )
                    st.write(correlation_analysis)
                except ValueError as ve:
                    st.warning(f"Error calculating correlations: {ve}")

                try:
                    outliers = pandas_agent.run(
                        "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis."
                    )
                    st.write(outliers)
                except ValueError as ve:
                    st.warning(f"Error identifying outliers: {ve}")

                try:
                    new_features = pandas_agent.run("What new features would be interesting to create?")
                    st.write(new_features)
                except ValueError as ve:
                    st.warning(f"Error suggesting new features: {ve}")

                return

            function_agent()

            # Custom linear regression model for specific selection
            def Random_forest():
                st.write("### Random Forest Model")

                # Using the uploaded dataset for model
                gold_data = df  # Assuming the uploaded CSV file contains a 'GLD' column

                # Plot distribution of 'GLD' price
                fig, ax = plt.subplots()
                sns.histplot(gold_data['GLD'], color='green', ax=ax)  # Updated to histplot for better clarity in Streamlit
                st.pyplot(fig)  # Passing the figure object to st.pyplot() to avoid the warning

                """Splitting the features and target"""
                X = gold_data.drop(['Date', 'GLD'], axis=1)
                Y = gold_data['GLD']

                """Splitting into Training and Test Data"""
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

                """Model Training: Random Forest Regressor"""
                regressor = RandomForestRegressor(n_estimators=100)
                regressor.fit(X_train, Y_train)

                # Prediction on Test Data
                test_data_prediction = regressor.predict(X_test)

                st.write(f"### Predicted values: {test_data_prediction}")

                # R squared error
                error_score = metrics.r2_score(Y_test, test_data_prediction)
                st.write('### R squared error:', error_score)

                """Compare the actual values and predicted values in a plot"""
                Y_test = list(Y_test)

                plt.plot(Y_test, color='blue', label='Actual Value')
                plt.plot(test_data_prediction, color='green', label='Predicted Value')
                plt.title('Actual vs Predicted Price')
                plt.xlabel('Number of values')
                plt.ylabel('GLD Price')
                plt.legend()
                
            '''def ann():
                st.write("### ANN Model")
                user_file = st.file_uploader("Upload your review dataset (CSV)", type=["csv"])
                
                if user_file is not None:
                    df = pd.read_csv(user_file)
                    
                    if 'Review Text' not in df.columns:
                        st.error("The uploaded dataset must contain a 'Review Text' column.")
                        return

                    # Create artificial sentiment labels
                    df['Sentiment'] = df['Review Text'].apply(
                        lambda x: 1 if any(word in x.lower() for word in ['good', 'amazing', 'excellent']) else 0
                    )

                    # Vectorize the text data using TF-IDF
                    tfidf = TfidfVectorizer(max_features=500)
                    X = tfidf.fit_transform(df['Review Text']).toarray()
                    y = df['Sentiment']

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Create and train the ANN model
                    model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)

                    # Display accuracy and classification report
                    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred) * 100:.2f}%")
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                    # Plot confusion matrix
                    fig, ax = plt.subplots()
                    conf_matrix = confusion_matrix(y_test, y_pred)
                    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)

                    # Plot loss curve if available
                    try:
                        fig, ax = plt.subplots()
                        ax.plot(model.loss_curve_)
                        ax.set_title("Loss Curve")
                        ax.set_xlabel("Epochs")
                        ax.set_ylabel("Loss")
                        st.pyplot(fig)
                    except AttributeError:
                        st.warning("Loss curve not available.")'''
            




            # Custom business problem and model selection using LangChain
            def chains_output(prompt, wiki_research):
                # Define prompt templates and chains
                data_problem_template = PromptTemplate(
                    input_variables=['business_problem'],
                    template='Convert the following business problem into a data science problem: {business_problem}.'
                )
                model_selection_template = PromptTemplate(
                    input_variables=['data_problem', 'wikipedia_research'],
                    template='Give a list of machine learning algorithms that are suitable to solve this problem: {data_problem}, while using this Wikipedia research: {wikipedia_research}.'
                )

                # Set up LangChain chains
                data_problem_chain = LLMChain(llm=llm, prompt=data_problem_template, verbose=True, output_key='data_problem')
                model_selection_chain = LLMChain(llm=llm, prompt=model_selection_template, verbose=True, output_key='model_selection')

                sequential_chain = SequentialChain(
                    chains=[data_problem_chain, model_selection_chain],
                    input_variables=['business_problem', 'wikipedia_research'],
                    output_variables=['data_problem', 'model_selection'],
                    verbose=True
                )

                my_chain_output = sequential_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
                my_data_problem = my_chain_output["data_problem"]
                my_model_selection = my_chain_output["model_selection"]
                return my_data_problem, my_model_selection

            # Main application logic for problem solving and model selection
            st.subheader('Data Science Problem')

            prompt = st.text_area('What is the business problem you would like to solve?')

            if prompt:
                wiki_research = WikipediaAPIWrapper().run(prompt)
                my_data_problem, my_model_selection = chains_output(prompt, wiki_research)

                st.write(my_data_problem)
                st.write(my_model_selection)

                # Machine Learning Algorithm selection
                formatted_list = ["Select Algorithm", "Random Forest", "Linear Regression"]
                selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

                if selected_algorithm == "Random Forest":
                    Random_forest()
                elif selected_algorithm == "Linear Regression":
                    st.write("### Random Forest Model Implementation Coming Soon!")
                    # Placeholder for Random Forest Model (API-based implementation)
                    pass

    with tab2:
        st.header("ChatBox")
        st.write("🤖 Welcome to the AI Assistant ChatBox!")
        st.write("Type in your queries, and the AI will assist you with insights and solutions for your data science project.")

        # Initialize chat history if not already in session state
        if 'responses' not in st.session_state:
            st.session_state['responses'] = ["How can I assist you?"]
        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        # Chat with OpenAI LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=apikey)
        if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

        # Define the prompt template
        system_msg_template = SystemMessagePromptTemplate.from_template(template="Answer truthfully using the context provided, or say 'I don't know' if unsure.")
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

        # Setup the conversation chain
        conversation = ConversationChain(
            memory=st.session_state.buffer_memory, 
            prompt=prompt_template, 
            llm=llm, 
            verbose=True
        )

        # Chat input and response container
        with st.container():
            query = st.text_input("How can I help you?", key="input")
            if query:
                with st.spinner("Thinking..."):
                    response = conversation.predict(input=query)
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

        # Display chat history
        for i in range(len(st.session_state['responses'])):
            # Display the assistant's response
            st.chat_message("assistant").write(st.session_state['responses'][i])
            
            # Display the user's request
            if i < len(st.session_state['requests']):
                st.chat_message("user").write(st.session_state['requests'][i])
