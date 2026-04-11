def get_bad_words():
    return ["puta", "putas", "caralho", "fodasse", "foda-se", "fodase", "paneleiro", "panilas", "merda", "merdas",
            "merdoso", "picha", "cona", "nigga", "nigger", "niger", "niga", "cabrao", "cabrão", "otário",
            "parvalhao", "parvalhão", "parvalhona", "chungoso", "bicha", "buceta", "otária", "panuco",
            "fse", "crl", "foder", "foda", "fodidos", "fodido", "fodida", "fodidas", "foderam", ]


def get_stop_words_nltk():
    # List of portuguese stop words from NLTK
    return ['a', 'à', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'até', 'com', 'como',
            'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele',
            'eles', 'em', 'entre', 'era', 'eram', 'éramos', 'essa', 'essas', 'esse', 'esses', 'esta', 'está', 'estamos',
            'estão', 'estar', 'estas', 'estava', 'estavam', 'estávamos', 'este', 'esteja', 'estejam', 'estejamos',
            'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem',
            'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora',
            'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'fui', 'há', 'haja', 'hajam',
            'hajamos', 'hão', 'havemos', 'haver', 'hei', 'houve', 'houvemos', 'houver', 'houvera', 'houverá',
            'houveram', 'houvéramos', 'houverão', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam',
            'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'lhe', 'lhes',
            'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no',
            'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas',
            'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são', 'se', 'seja', 'sejam', 'sejamos', 'sem',
            'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'seu', 'seus', 'só', 'somos',
            'sou', 'sua', 'suas', 'também', 'te', 'tem', 'tém', 'temos', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá',
            'terão', 'terei', 'teremos', 'teria', 'teriam', 'teríamos', 'teu', 'teus', 'teve', 'tinha', 'tinham',
            'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos', 'tivesse',
            'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'vocês', 'vos']


def get_stop_words_fineweb2():
    # List of portuguese stop words from FineWeb2 dataset
    return [
        "de", "a", "e", "o", "em", "do", "da", "que", "um", "no", "uma",
        "com", "para", "na", "é", "foi"
    ]
