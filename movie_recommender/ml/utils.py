def is_ascii(text):
    try:
        test = text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False
