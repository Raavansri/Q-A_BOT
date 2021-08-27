from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request
from newspaper import Article
import numpy as np
import string
import random
import nltk

nltk.download('punkt')

app=Flask('Purpple')
@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def ping():
    test = "The earliest known soil classification system comes from China, appearing in the book Yu Gong (5th century BCE), where the soil was divided into three categories and nine classes, depending on its color, texture and hydrology.Contemporaries Friedrich Albert Fallou, the German founder of modern soil science, and Vasily Dokuchaev, the Russian founder of modern soil science, are both credited with being among the first to identify soil as a resource whose distinctness and complexity deserved to be separated conceptually from geology and crop production and treated as a whole. As a founding father of soil science Fallou has primacy in time. Fallou was working on the origins of soil before Dokuchaev was born, however Dokuchaev's work was more extensive and is considered to be the more significant to modern soil theory than Fallou's.Previously, soil had been considered a product of chemical transformations of rocks, a dead substrate from which plants derive nutritious elements. Soil and bedrock were in fact equated. Dokuchaev considers the soil as a natural body having its own genesis and its own history of development, a body with complex and multiform processes taking place within it. The soil is considered as different from bedrock. The latter becomes soil under the influence of a series of soil-formation factors (climate, vegetation, country, relief and age). According to him, soil should be called the daily or outward horizons of rocks regardless of the type; they are changed naturally by the common effect of water, air and various kinds of living and dead organisms.A 1914 encyclopedic definition:the different forms of earth on the surface of the rocks, formed by the breaking down or weathering of rocks. serves to illustrate the historic view of soil which persisted from the 19th century. Dokuchaev's late 19th century soil concept developed in the 20th century to one of soil as earthy material that has been altered by living processes. A corollary concept is that soil without a living component is simply a part of earth's outer layer.Further refinement of the soil concept is occurring in view of an appreciation of energy transport and transformation within soil. The term is popularly applied to the material on the surface of the Earth's moon and Mars, a usage acceptable within a portion of the scientific community. Accurate to this modern understanding of soil is Nikiforoff's 1959 definition of soil as the excited skin of the sub aerial part of the earth's crust.Areas of practice Academically, soil scientists tend to be drawn to one of five areas of specialization: microbiology, pedology, edaphology, physics, or chemistry. Yet the work specifics are very much dictated by the challenges facing our civilization's desire to sustain the land that supports it, and the distinctions between the sub-disciplines of soil science often blur in the process. Soil science professionals commonly stay current in soil chemistry, soil physics, soil microbiology, pedology, and applied soil science in related disciplines One interesting effort drawing in soil scientists in the USA as of 2004 is the Soil Quality Initiative. Central to the Soil Quality Initiative is developing indices of soil health and then monitoring them in a way that gives us long term (decade-to-decade) feedback on our performance as stewards of the planet. The effort includes understanding the functions of soil microbiotic crusts and exploring the potential to sequester atmospheric carbon in soil organic matter. The concept of agriculture in relation to soil quality, however, has not been without its share of controversy and criticism, including critiques by Nobel Laureate Norman Borlaug and World Food Prize Winner Pedro Sanchez.A more traditional role for soil scientists has been to map soils. Most every area in the United States now has a published soil survey, which includes interpretive tables as to how soil properties support or limit activities and uses. An internationally accepted soil taxonomy allows uniform communication of soil characteristics and soil functions. National and international soil survey efforts have given the profession unique insights into landscape scale functions."
    sentence_list = nltk.sent_tokenize(test)

    def gretting(text):
        text = text.lower()

        greeting_bot = ['HI', 'HELLOW', 'HEY THERE', 'WELCOME']
        greeting_user = ['hellow', 'hi', 'is there any one', 'hay', 'hey']

        exit_list = ['see you later', 'bye', 'thank you']
        exit_responce='you welocme... if wanna quit click quit button! '

        for word in text.split():
            if word in greeting_user:
                wish= random.choice(greeting_bot)
                return wish
            if word in exit_list:
                wish= exit_responce
                return wish

    def index_sort(list_var):
        length = len(list_var)
        list_index = list(range(0, length))

        x = list_var
        for i in range(length):
            for j in range(length):
                if x[list_index[i]] > x[list_index[j]]:
                    temp = list_index[i]
                    list_index[i] = list_index[j]
                    list_index[j] = temp

        return list_index

    def response(user_input):
        user_input = user_input.lower()
        sentence_list.append(user_input)
        response = ' '
        cm = CountVectorizer().fit_transform(sentence_list)
        similarity_scores = cosine_similarity(cm[-1], cm)
        similarity_score_list = similarity_scores.flatten()
        index = index_sort(similarity_score_list)
        index = index[1:]
        response_flag = 0

        j = 0

        for i in range(len(index)):
            if similarity_score_list[index[i]] > 0.0:
                response = response + ' ' + sentence_list[index[i]]
                response_flag = 1
                j = j + 1
            if j > 2:
                break

        if response_flag == 0:
            response = response + ' ' + "It's out of my understanding,from document!"

        sentence_list.remove(user_input)

        return response

    user_input = request.form['usr_inpt']

    if gretting(user_input) != None:
        send='purple:' + gretting(user_input)
    else:
        send='purple:' + response(user_input)

    return render_template('home.html', send_text=send)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)

