# usual libraries and app
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import openai

# AI app
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import streamlit.components.v1 as components


# default params
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
BASE_FOLDER = "/mount/src/city-app/"

# create the app
st.title('City of Calgary Tweet Assessment')
st.caption('A Tool to Automatically Identify Tone of Voice and Topics Relating to the City')

st.write("""
In numerous cities, population expansion and technological advancements necessitate proactive modernization and integration of technology. However, the existing bureaucratic structure often hinders local officials' efforts to effectively address and monitor residents' needs and enhance the city accordingly. Understanding what people find important and useful can be inferred from their posts on social media. Twitter, as one of the most popular social media platforms, provides us with valuable data that, with the right tools and analysis, can provide insights into the performance of urban services and residents' perception of them. 
In this study, we used the city of Calgary as an exemplar to gather tweets and analyze topics relating to city development, urban planning, and minorities.
 Natural language processing (NLP) techniques were used and developed to preprocess stored tweets, classify the emotions, and identify the topics present in the dataset to eventually provide a set of topics with the prevalent emotion in that topic. 
We utilized a variety of methods to analyze the collected data. BERTopic for topic modeling and few-shot learning using Setfit for emotion analysis outperformed the others. Hence, we identify issues related to city development, senior citizens, taxes, and unemployment using these methods, and we demonstrate how delving into these analyses can improve urban planning.     
""")

link = 'Please visit our [GitHub](https://github.com/mitramir55/Advancing-smart-cities-with-nlp) if you want to see the code or make any contributions.'
st.markdown(link, unsafe_allow_html=True)

link = 'If you use this package, please cite it as: ....'
st.markdown(link, unsafe_allow_html=True)

#opening the image
image = Image.open(BASE_FOLDER + 'static files/pic_overall_approach.png')

#displaying the image on streamlit app
st.image(image, caption='Overall approach')

st.title("Try our algorithm")
# st.caption("Disclaimer: This website is provided solely for the purpose of testing this package. As a result, it can handle a few thousand (up to 10,000) one-line sentences or a few hundred paragraphs in each file.")

# single text
st.subheader("Single sentence analysis")

st.write('the path to the file ', os.path.abspath(".env"))

def bertopic_model():

    loaded_model = BERTopic.load(BASE_FOLDER + "BERTopic")
    return loaded_model

def predict_label(input_sent):
    # do the prediction
    predicted_topics, predicted_probs = bertopic_m.transform(documents=[input_sent])
    chosen_topic = predicted_topics[0]
    confidence = predicted_probs[0]
    return chosen_topic, confidence

def get_topic_rep_and_name(chosen_topic):

    topic_df = bertopic_m.get_topic_info()#.head(30)
    topic_row_info = topic_df[topic_df.loc[:, 'Topic']==chosen_topic]
    topic_name = [i for i in topic_row_info.Name.values][0]
    topic_rep = [i for i in topic_row_info.Representation.values][0]

    return topic_name, topic_rep

def read_prompt_df(first_n):
    """
    reads the data that will help us with creating the prompt
    """

    url = "https://raw.githubusercontent.com/mitramir55/Advancing-smart-cities-with-nlp/main/Dataset/split%2020-80/training_set.csv"
    df = pd.read_csv(url, nrows=first_n)

    labels_to_int = {'anger': 0, 'joy': 1, 'optimism': 2, 'sadness': 3}
    int_to_label = {v:k for (k, v) in labels_to_int.items()}
    def emotion_labeler(label_id):
        return int_to_label[label_id]

    # build the textual label
    df.loc[:, 'emotion'] = df.loc[:, 'label'].apply(lambda x: emotion_labeler(x))

    return df


# single sentence 
input_sent = st.text_input("Please enter your sentence in the box bellow:", "")
button_1 = st.button('analyze Emotion', key='butt1')
button_2 = st.button('analyze Topic', key='butt2')

# sentiment analysis
if input_sent and button_1:

    st.write('Building the model...')

    # read the dataset for sentiment analysis as the few-shot input
    df = read_prompt_df(first_n=10)

    prompt_prefix_1 = """
    You are a helpful assistant that classifies text as having the following 
    emotions: anger, joy, optimism, and sadness.\n"""


    # build the prompt
    prompt_prefix_2 = ""
    for i in range(len(df)):
        prompt_prefix_2 += (f"Text: '{df.loc[i, 'preprocessed']}' - " +\
            f"Emotion: '{df.loc[i, 'emotion']}' \n")
        

    prompt_prefix = prompt_prefix_1 + prompt_prefix_2

    # create kernel for semantic kernel
    kernel = sk.Kernel()

    print()
    # Prepare OpenAI service using credentials stored in the `.env` file
    os.cd(os.path.dirname(BASE_FOLDER))

    # api_key, org_id = sk.openai_settings_from_dot_env()

    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path= '/mount/src/city-app/.env')

    api_key = os.getenv('OPENAI_API_KEY')
    org_id = None
    print('api_key = ', api_key)

    kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4", api_key, org_id))

    prompt = f"{prompt_prefix}\n" + f"Text is: {input_sent} - " + "what is the Emotion: "
    classifier = kernel.create_semantic_function(prompt)

    # Summarize the list
    summary_result = classifier(prompt)
    st.write("GPT-4 few-shot model says this represents ", summary_result)



    openai.api_key_path = '/mount/src/city-app/.env'
 # Topic modeling ----------------------------------   


# BERTopic
from bertopic import BERTopic
###### Topic modeling
bertopic_m = bertopic_model()

if input_sent and button_2:

    chosen_topic, confidence = predict_label(input_sent)
    topic_name, topic_rep = get_topic_rep_and_name(chosen_topic)

    st.write(f"""
            The topic chosen for this sentence is topic number {chosen_topic}.\n
            The topic representation is {topic_rep}.\n
            """)


#import plotly.offline as pyo
# fig = bertopic_m.visualize_topics()
# Export the graph to HTML
# pyo.plot(fig, filename="C:\Users\mitra\city-app\BERTopic")
vis_path = BASE_FOLDER + "static files/viz.html"
HtmlFile = open(vis_path, 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, height = 600)





#<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>
# st.components.v1.iframe(vis_path, height=600)
