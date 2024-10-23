from PIL import Image
import streamlit as st
import  configparser
import os
from functools import wraps
from time import time


def add_logo(logo_path, width, height):
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo
