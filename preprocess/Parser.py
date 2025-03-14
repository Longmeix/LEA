import re
from enum import Enum
from typing import List, Tuple

from tools.MultiprocessingTool import MultiprocessingTool

pref = {
    "rdf:": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs:": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd:": "http://www.w3.org/2001/XMLSchema#",
    "owl:": "http://www.w3.org/2002/07/owl#",
    "pos:": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "skos:": "http://www.w3.org/2004/02/skos/core#",
    "dc:": "http://purl.org/dc/terms/",
    "foaf:": "http://xmlns.com/foaf/0.1/",
    "vcard:": "http://www.w3.org/2006/vcard/ns#",
    "dbp:": "http://dbpedia.org/property/",
    "dbr:": "http://dbpedia.org/resource/",
    "ontology:": "http://dbpedia.org/ontology/",
    "db:": "http://dbpedia.org/",
    "y1:": "http://www.mpii.de/yago/resource/",
    "y2:": "http://yago-knowledge.org/resource/",
    "geo:": "http://www.geonames.org/ontology#",
    'wikie:': 'http://www.wikidata.org/entity/',
    'wiki:': 'http://www.wikidata.org/',
    'schema:': 'http://schema.org/',
    'fb2:': 'http://rdf.freebase.com/ns/',
    'freebase:': 'http://rdf.freebase.com/',
    'dbp_zhr': 'http://zh.dbpedia.org/resource/',
    'dbp_zhp': 'http://zh.dbpedia.org/property/',
    'dbp_zh': 'http://zh.dbpedia.org/',
    'dbp_frr': 'http://fr.dbpedia.org/resource/',
    'dbp_frp': 'http://fr.dbpedia.org/property/',
    'dbp_fr': 'http://fr.dbpedia.org/',
    'dbp_jar': 'http://ja.dbpedia.org/resource/',
    'dbp_jap': 'http://ja.dbpedia.org/property/',
    'dbp_ja': 'http://ja.dbpedia.org/',
    'dbp_der': 'http://de.dbpedia.org/resource/',
    'dbp_dep': 'http://de.dbpedia.org/property/',
    'dbp_de': 'http://de.dbpedia.org/',
    'purl': 'http://purl.org/dc/elements/1.1/',
    'xmls': 'http://xmlns.com/foaf/0.1/',
    'low_zh': 'http://zh.dbpedia.org/resource/',
    'low_vi': 'http://vi.dbpedia.org/resource/',
    'low_th': 'http://th.dbpedia.org/resource/',
    "en_wiki:": "https://en.wikipedia.org/wiki/",  # icews_wiki
    "doremus": "http://data.doremus.org/",  # doremus_en
    "erlangen": "http://erlangen-crm.org/",
    "isni": "http://isni.org/",
}


class OEAFileType(Enum):
    attr = 0
    rel = 1
    ttl_full = 2
    truth = 3
    rel_id = 4
    ent_name = 5


def strip_square_brackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s


def compress_uri(uri):
    uri = strip_square_brackets(uri)
    if uri.startswith("http://"):
        for key, val in pref.items():
            if uri.startswith(val):
                uri = uri.replace(val, '')
                # uri = uri.replace("_", " ").replace(u'\xa0', '')  # 删除下划线
                uri = uri.replace("_", " ").replace("%", "").replace('\t', ' ').replace(u'\xa0', '')
    return uri


def oea_attr_line(line: str):
    fact: List[str] = line.strip('\n').split('\t')
    if not fact[2].startswith('"'):
        fact[2] = ''.join(('"', fact[2], '"'))
    return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])


def oea_rel_line(line: str) -> Tuple:
    fact: List[str] = line.strip().split('\t')
    try:
        return compress_uri(fact[0]), compress_uri(fact[1]), compress_uri(fact[2])
    except Exception as e:
        print(e)
        print(line)
        print(fact)

# def oea_rel_id_line(line: str) -> Tuple:
#     fact: List[int] = [int(i) for i in line.strip().split('\t')]


def oea_truth_line(line: str) -> Tuple:
    fact: List[str] = line.strip().split('\t')
    return compress_uri(fact[0]), compress_uri(fact[1])

def oea_ent_name_line(line: str) -> Tuple:
    fact: List[str] = line.strip().split('\t')
    return int(fact[0]), compress_uri(fact[-1])




def stripSquareBrackets(s):
    # s = ""
    if s.startswith('"'):
        rindex = s.rfind('"')
        if rindex > 0:
            s = s[:rindex + 1]
    else:
        if s.startswith('<'):
            s = s[1:]
        if s.endswith('>'):
            s = s[:-1]
    return s

ttlPattern = "([^\\s]+)\\s+([^\\s]+)\\s+(.+)\\s*."  # zh_en, zh_vi, doremus_en...
# ttlPattern = r"([^\t]+)\t([^\t]+)\t(.+)"

def ttl_no_compress_line(line):
    if line.startswith('#'):
        return None, None, None
    fact = re.match(ttlPattern, line.rstrip())
    if fact is None:
        # print(line)
        return 'Error'
    sbj = stripSquareBrackets(fact[1])
    pred = stripSquareBrackets(fact[2])
    obj = stripSquareBrackets(fact[3])
    return sbj, pred, obj


def for_file(file, file_type: OEAFileType) -> list:
    line_solver = None
    if file_type == OEAFileType.attr:
        line_solver = oea_attr_line
    elif file_type == OEAFileType.rel:
        line_solver = oea_rel_line
    elif file_type == OEAFileType.ttl_full:
        line_solver = ttl_no_compress_line
    elif file_type == OEAFileType.truth:
        line_solver = oea_truth_line
    elif file_type == OEAFileType.ent_name:
        line_solver = oea_ent_name_line
    # elif file_type == OEAFileType.rel_id: # for low resource language data (zh_vi, ..)
    #     line_solver = oea_rel_id_line
    assert line_solver is not None
    with open(file, 'r', encoding='utf-8') as rfile:
        mt = MultiprocessingTool()
        results = mt.packed_solver(line_solver).send_packs(rfile).receive_results()
        # results = [line_solver(line) for line in rfile]
        results = [triple for triple in results if triple[0] is not None]
    return results
