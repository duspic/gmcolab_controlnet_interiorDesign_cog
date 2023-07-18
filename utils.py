def style_to_prompt(style:str):
    dct = {'scandinavian':"nordic, clean, scandinavian modern ikea style",
            'rustic':"cottage rustic traditional style, cozy, warm",
            'bohemian':"bohemian, boho style, cozy comfy, carpets, plants, books ",
            'industrial':"industrial style, concrete floor, steel and timber, "
            }
    
    return dct[style]