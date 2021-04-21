from google_trans_new import google_translator


def tr2english(text, user_data):
    translator = google_translator()
    msg_tr = translator.translate(text=text)
    if 'language' not in user_data:
        user_data['language'] = translator.detect(text)[0]
        print(user_data['language'])
    return msg_tr


def tr2other(text, language):
    translator = google_translator()
    msg_tr = translator.translate(text, lang_tgt=str(language))
    return msg_tr
