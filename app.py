# usual libraries and app
import streamlit as st
import pandas as pd
import numpy as np

# BERTopic
from bertopic import BERTopic

# AI app
from pathlib import Path
import json
from collections import defaultdict
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion


# default params
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


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


st.title("Try our algorithm")
# st.caption("Disclaimer: This website is provided solely for the purpose of testing this package. As a result, it can handle a few thousand (up to 10,000) one-line sentences or a few hundred paragraphs in each file.")



# single text
st.subheader("Single sentence analysis")

st.write(f'path is = {Path.cwd()}')

def bertopic_model():

    loaded_model = BERTopic.load("city-app\\BERTopic")
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




# single sentence 
input_sent = st.text_input("Please enter your sentence in the box bellow:", "")
button_1 = st.button('analyze', key='butt1')



###### Topic modeling
bertopic_m = bertopic_model()


chosen_topic, confidence = predict_label(input_sent)
topic_name, topic_rep = get_topic_rep_and_name(chosen_topic)

st.write("""
         The topic chosen for this sentence is topic number {chosen_topic}.\n
        The topic representation is {topic_rep}.\n
         
         """)

bertopic_m.visualize_topics()
#<iframe src="viz.html" style="width:1000px; height: 680px; border: 0px;""></iframe>
st.components.v1.iframe(iframe_url="viz.html", height=600)

    
bertopic_model()