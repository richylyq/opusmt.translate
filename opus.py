"""
translation program for simple text
1. detect language from langdetect
2. translate to target language given by user

Example from
https://www.thepythoncode.com/article/machine-translation-using-huggingface-transformers-in-python 

user_input:
    string: string to be translated
    target_lang: language to be translated to

Returns:
    string: translated string of text
"""

import argparse

from langdetect import DetectorFactory

DetectorFactory.seed = 0

from langdetect import detect
from transformers import pipeline


def detect_lang(article, target_lang):
    """
    Language Detection using library langdetect

    Args:
        article (string): article that user wish to translate
        target_lang (string): language user want to translate article into

    Returns:
        string: detected language short form
    """
    result_lang = detect(article)

    if result_lang == target_lang:
        return result_lang
    else:
        return result_lang


def opus_trans(article, result_lang, target_lang):
    """
    Translation by Helsinki-NLP model

    Args:
        article (string): article that user wishes to translate
        result_lang (string): detected language in short form
        target_lang (string): language that user wishes to translate article into

    Returns:
        string: translated piece of article based off target_lang
    """

    task_name = f"translation_{result_lang}_to_{target_lang}"
    model_name = f"Helsinki-NLP/opus-mt-{result_lang}-{target_lang}"
    translator = pipeline(task_name, model=model_name, tokenizer=model_name)
    translated = translator(article)[0]["translation_text"]
    print(translated)
    return translated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--article",
        help="Upload the article you wish to translate",
        required=True,
        dest="article",
    )
    parser.add_argument(
        "-t",
        "--target_lang",
        help="Choose your desired language to translate to",
        required=True,
        dest="target_lang",
    )

    args = parser.parse_args()
    result_lang = detect_lang(args.article, args.target_lang)

    if result_lang == "zh-cn":
        result_lang = "zh"
    elif result_lang == "zh-tw":
        result_lang = "zh"

    if result_lang == args.target_lang:
        pass
    else:
        opus_trans(args.article, result_lang, args.target_lang)
