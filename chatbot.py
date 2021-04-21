# telegram api
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters, CommandHandler
from telegram import ReplyKeyboardRemove

from traductor import *
from treatLocation import *
from treatInput import *
from freq import *

# nltk
import nltk
import numpy as np
import random
import string  # to process standard python strings
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import pandas as pd
import os
from staticmap import StaticMap, CircleMarker
from math import *

# ******************************************************************
# GREETINGS
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# ******************************************************************
# DICTIONARY OF TYPES OF MELANOMA

tipus = [
    'uveal melanoma',
    'diffuse leptomeningeal melanocytosis',
    'familial atypical multiple mole melanoma syndrome',
    'familial melanoma',
    'malignant melanoma of the mucosa',
    'melanoma and neural system tumor syndrome',
    'melanoma of soft tissue',
    'primary melanoma of the central nervous system'
]
names = ['uveal melanoma', 'primary melanoma of the central nervous system', 'melanoma of soft tissue',
         'diffuse leptomeningeal melanocytosis', 'melanoma and neural system tumor syndrome',
         'familial atypical multiple mole melanoma syndrome', 'familial melanoma', 'malignant melanoma of the mucosa']

sinonims = [
    ['choroidal melanoma', 'iris melanoma', 'Intraocular melanoma'],
    ['DLM Leptomeningeal melanomatosis'],
    ['B-K mole syndrome',
     'FAMM-PC syndrome',
     'FAMMM syndrome',
     'Familial Clark nevus syndrome',
     'Familial atypical mole syndrome',
     'Familial atypical multiple mole melanoma-pancreatic carcinoma syndrome',
     'Familial dysplastic nevus syndrome',
     'Melanoma-pancreatic cancer syndrome'],
    ['Dysplastic nevus syndrome hereditary', 'B-K mole syndrome'],
    [],
    ['Melanoma-astrocytoma syndrome'],
    ['Clear cell sarcoma of the tendons and aponeuroses'],
    ['Malignant melanoma of meninges', 'Primary melanoma of the CNS']
]

dic = {}
types_melanomes = []
for i in range(0, len(sinonims)):
    family = sinonims[i]
    type = tipus[i].lower()
    dic[type.lower()] = type.lower()
    for sin in family:
        dic[sin.lower()] = type.lower()
        types_melanomes.append(sin.lower())


# *********************************************************************
# WE WILL USE THIS TO READ THE INFORMATION OF THE DISEASE WE HAVE FOUND

def readCSV(name):
    global information_hospitals
    information_hospitals = []

    print(name)
    index = names.index(str(name)) + 1
    print(str(index))
    name_file = 'csv/drugs_labs_biobanks_dataset -' + str(index) + '.csv'
    print('b')
    rows = pd.read_csv('Intraocular_melanoma.csv').values
    print('a')
    for row in rows:
        name = row[4]
        telefon = row[5]
        address = row[6]
        lat = row[7]
        long = row[8]
        if (not pd.isnull(name)) and (not pd.isnull(telefon)) and (not pd.isnull(address)) and (
                not pd.isnull(lat)) and (not pd.isnull(long)):
            information_hospitals.append([name, telefon, address, lat, long])


# ******************************************************************

# function to be called
def answer(update, context):
    user_response = tr2english(update.message.text, context.user_data).lower()
    words = word_tokenize(user_response)
    if user_response == 'bye' or user_response == 'Bye':
        update.message.reply_text(tr2other("Goodbye!!", context.user_data['language']))

    elif user_response == 'thanks' or user_response == 'thank you':
        update.message.reply_text(tr2other("You are very welcome", context.user_data['language']))

    elif greeting(user_response) is not None:  # when greetings appear
        text = greeting(
            user_response) + " " + update.message.chat.first_name + "\nIt is very important for me to make sure that you have been already diagnosed by a professional. Could you please confirm me so?"
        update.message.reply_text(tr2other(text, context.user_data['language']))

    elif 'diagnosed' not in context.user_data:
        if ('no' in words) or ("n't" in words) or ("not" in words):
            update.message.reply_text(tr2other(
                "Don't worry. In this case we believe you should first go to a certified center to diagnose the exact type of melanoma that you suffer",
                context.user_data['language']))

        else:
            context.user_data['diagnosed'] = True
            update.message.reply_text(tr2other(
                "Then, it would be very helpful for me to know what specific type of Melanoma you are interested in. Do you know its name?",
                context.user_data['language']))


    elif ('type_disease' not in context.user_data) and ('diagnosed' in context.user_data):
        if ('no' in words) or ("n't" in words) or ("not" in words):
            update.message.reply_text(tr2other("Don't worry...", context.user_data['language']))

        else:
            found = False
            for type in types_melanomes:
                if type in user_response:
                    found = True
                    context.user_data['type_disease'] = dic[type]
                    readCSV(context.user_data['type_disease'])
                    print(found)

            if found:
                update.message.reply_text(tr2other("Perfect", context.user_data['language']))
                update.message.reply_text(tr2other(
                    "Now, it is important for me to know what drugs have been prescribed for this disease by your doctor?",
                    context.user_data['language']))

            else:
                update.message.reply_text(tr2other("Go ahead and tell me what is its name, please.", context.user_data['language']))


    elif 'medicin' not in context.user_data:
        update.message.reply_text("Thank you. This will be very helpful for me to know you a little better. Is there anything I can help you with?")
        context.user_data['medicin'] = user_response


    else:
        words = treatInput(user_response)
        words = " ".join(words)
        print(words)
        prediction = predictAnswer(words)
        print(prediction)

        if words[0] == '$' or ('where' in words):
            update.message.reply_text(tr2other("Please send me your location, so I can give you the best option.",
                                               context.user_data['language']))

        elif words[0] == 'is':
            update.message.reply_text(tr2other("Yes, it is"))

        elif words[0] == 'should':
            update.message.reply_text(tr2other("Yes, you should"))

        elif words[0] == 'would':
            update.message.reply_text(tr2other("Yes, I would"))

        elif words[0] == 'does':
            update.message.reply_text(tr2other("Yes, it does"))

        elif words[0] == 'do':
            update.message.reply_text(tr2other("Yes, you do"))

        else:
            update.message.reply_text(prediction)


def findClosestHospital(lat, long):
    list = []
    for i in range(0, len(information_hospitals)):
        hospital = information_hospitals[i]
        lat2 = hospital[3]
        long2 = hospital[4]
        distance = distancia([lat, long], [lat2, long2])

        list.append([distance, i])

    list = sorted(list, key=lambda x: x[0])
    return list[0][1]


def giveClosestHospital(update, context):
    try:
        lat = str(update.message.location.latitude)
        lon = str(update.message.location.longitude)
        index = findClosestHospital(lat, lon)
        entry = information_hospitals[index]
        nom = entry[0]
        telefon = entry[1]
        address = entry[2]
        lat = entry[3]
        lon = entry[4]

        update.send_location(chat_id=update.message.chat_id, latitude=lat, longitude=lon)

        text = "This is the location of the nearest hospital in which your type of disease can be treated.\nThis hospital is called " + nom + " and its address is " + address + "\n" + "I would recommend you to contact it through the telefon " + telefon + " so you can book a meet with an specialist."
        update.message.reply_text(tr2other(text, context.user_data['language']))
    except Exception as e:
        print(e)
    #   bot.send_message(chat_id=update.message.chat_id,text='Something went wrong! Please try it again later.')


def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Welcome to RareHackBot! Ask me.')


def end(update, context):
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())
    updater.stop()  # end bot


if __name__ == '__main__':
    # load the access token
    TOKEN = open('token.txt').read().strip()

    updater = Updater(token=TOKEN)
    dispatcher = updater.dispatcher

    # handling the call
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('end', end))
    dispatcher.add_handler(MessageHandler(Filters.text, answer))
    dispatcher.add_handler(MessageHandler(Filters.location, giveClosestHospital))  # telegram api

    # starting the bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    # updater.idle()
